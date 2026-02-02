import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob
import argparse
from tqdm import tqdm

from models import StateAutoencoder, LatentDynamicsModel, dynamics_loss

LATENT_DIM = 32
MODEL_DIR = 'saved_models'
RESULT_DIR = 'results'


def modify_low_gravity(z_next, action):
    is_jump = (action == 2) | (action == 3)
    scale = torch.where(is_jump.unsqueeze(1),
                        torch.ones_like(z_next) * 1.4,
                        torch.ones_like(z_next))
    return z_next * scale


def modify_normal(z_next, action):
    return z_next


def train_dual_dynamics(z_all, actions, z_next_all, device):
    print("7 dual")
    model1 = LatentDynamicsModel(LATENT_DIM, action_dim=4).to(device)
    model2 = LatentDynamicsModel(LATENT_DIM, action_dim=4).to(device)
    opt1 = optim.Adam(model1.parameters(), lr=1e-3)
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(z_all, actions, z_next_all),
        batch_size=128, shuffle=True
    )
    for epoch in range(30):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/30", leave=False):
            z, a, z_next = [x.to(device) for x in batch]
            pred1 = model1(z, a)
            loss1 = dynamics_loss(pred1, modify_normal(z_next, a))
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            pred2 = model2(z, a)
            loss2 = dynamics_loss(pred2, modify_low_gravity(z_next, a))
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        if (epoch + 1) % 10 == 0:
            print(epoch+1, loss1.item(), loss2.item())
    torch.save(model1.state_dict(), os.path.join(MODEL_DIR, 'dynamics_normal.pt'))
    torch.save(model2.state_dict(), os.path.join(MODEL_DIR, 'dynamics_lowgrav.pt'))
    return model1, model2


def blended_rollout(model1, model2, z0, actions, alpha, device):
    traj = [z0.unsqueeze(0)]
    z = z0.unsqueeze(0)
    for a in actions:
        a_tensor = torch.tensor([a], device=device)
        with torch.no_grad():
            z1 = model1(z, a_tensor)
            z2 = model2(z, a_tensor)
        z = alpha * z1 + (1 - alpha) * z2
        traj.append(z)
    return torch.cat(traj, dim=0)


def visualize_blending(ae, model1, model2, z0, device, num_steps=12):
    print("blend viz")
    actions = [3] * num_steps
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    view_size = int(round(np.sqrt(ae.state_dim)))
    if view_size * view_size != ae.state_dim:
        view_size = 13
    fig, axes = plt.subplots(len(alphas), num_steps + 1, figsize=(2*(num_steps+1), 2*len(alphas)))
    for row, alpha in enumerate(alphas):
        traj = blended_rollout(model1, model2, z0, actions, alpha, device)
        with torch.no_grad():
            decoded = ae.decode(traj)
        for col in range(num_steps + 1):
            ax = axes[row, col]
            img = decoded[col].cpu().numpy().reshape(view_size, view_size)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f'a={alpha:.2f}', fontsize=10, rotation=0, ha='right')
            if row == 0:
                ax.set_title(f't={col}', fontsize=9)
    plt.suptitle('blend', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'physics_blending.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("blend ok")


def visualize_blending_trajectories(model1, model2, z0, device, num_steps=20):
    actions = [3] * num_steps
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_trajs = []
    for alpha in alphas:
        traj = blended_rollout(model1, model2, z0, actions, alpha, device)
        all_trajs.append(traj.cpu().numpy())
    all_points = np.vstack(all_trajs)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)
    trajs_2d = []
    idx = 0
    for traj in all_trajs:
        trajs_2d.append(all_2d[idx:idx+len(traj)])
        idx += len(traj)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    for i, (alpha, traj_2d) in enumerate(zip(alphas, trajs_2d)):
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'o-', color=colors[i],
                label=f'a={alpha:.2f}', markersize=6, linewidth=2, alpha=0.8)
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=colors[i], s=150, marker='s', edgecolor='black')
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=colors[i], s=150, marker='^', edgecolor='black')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('blend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'blending_trajectories.png'), dpi=150)
    plt.close()
    print("blend traj ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    if args.data is None:
        candidates = sorted(glob.glob(os.path.join(ROOT, 'data', 'mario_data_*.npz')), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("no npz")
            return
        args.data = candidates[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)
    os.makedirs(RESULT_DIR, exist_ok=True)

    data = np.load(args.data)
    states = torch.FloatTensor(data['states'])
    actions = torch.LongTensor(data['actions'])
    next_states = torch.FloatTensor(data['next_states'])
    state_dim = states.shape[1]

    ae = StateAutoencoder(state_dim, LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'autoencoder.pt'), map_location=device))
    ae.eval()

    print("encode")
    with torch.no_grad():
        z_all, z_next_all = [], []
        for i in range(0, len(states), 1024):
            s = states[i:i+1024].to(device)
            s_next = next_states[i:i+1024].to(device)
            _, z = ae(s)
            _, z_next = ae(s_next)
            z_all.append(z.cpu())
            z_next_all.append(z_next.cpu())
        z_all = torch.cat(z_all)
        z_next_all = torch.cat(z_next_all)

    model1, model2 = train_dual_dynamics(z_all, actions, z_next_all, device)
    z0 = z_all[100].to(device)
    visualize_blending(ae, model1, model2, z0, device)
    visualize_blending_trajectories(model1, model2, z0, device)
    print("done")


if __name__ == '__main__':
    main()
