import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob
import argparse

from models import StateAutoencoder, LatentDynamicsModel

LATENT_DIM = 32
RESULT_DIR = 'results'
MODEL_DIR = 'saved_models'
ACTION_NAMES = ['NOOP', 'RIGHT', 'JUMP', 'RIGHT+JUMP']


def evaluate_rollout(ae, dyn, states, actions, next_states, device, num_steps=10):
    print("5.1 rollout", num_steps)
    ae.eval()
    dyn.eval()
    view_size = int(round(np.sqrt(states.shape[1])))
    if view_size * view_size != states.shape[1]:
        view_size = 13
    max_start = max(0, len(states) - num_steps - 2)
    if max_start < 4:
        indices = list(range(min(4, max_start + 1)))
    else:
        indices = np.random.choice(max_start + 1, min(4, max_start + 1), replace=False)
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()
    fig, axes = plt.subplots(4, num_steps + 1, figsize=(2*(num_steps+1), 8))
    for row, idx in enumerate(indices):
        if idx + num_steps + 1 > len(states):
            idx = max(0, len(states) - num_steps - 1)
        real_seq = states[idx:idx+num_steps+1].to(device)
        real_actions = actions[idx:idx+num_steps]
        with torch.no_grad():
            _, z = ae(real_seq[0:1])
        z_traj = [z]
        for i in range(num_steps):
            with torch.no_grad():
                z = dyn(z, real_actions[i:i+1].to(device))
            z_traj.append(z)
        z_traj = torch.cat(z_traj, dim=0)
        with torch.no_grad():
            decoded = ae.decode(z_traj)
        for col in range(num_steps + 1):
            ax = axes[row, col]
            real_img = real_seq[col].cpu().numpy().reshape(view_size, view_size)
            pred_img = decoded[col].cpu().numpy().reshape(view_size, view_size)
            combined = np.vstack([real_img, np.ones((1, view_size))*0.5, pred_img])
            ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title('t=0' if col == 0 else f't={col}\n{ACTION_NAMES[real_actions[col-1].item()]}', fontsize=8)
    plt.suptitle('rollout', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'rollout_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("rollout ok")


def evaluate_action_trajectories(ae, dyn, states, device, num_steps=15):
    print("5.2 traj")
    idx = np.random.randint(100, max(101, len(states) - 100))
    s0 = states[idx:idx+1].to(device)
    with torch.no_grad():
        _, z0 = ae(s0)
    trajectories = {}
    for action_id in range(4):
        z = z0.clone()
        traj = [z.cpu().numpy()]
        for _ in range(num_steps):
            with torch.no_grad():
                z = dyn(z, torch.tensor([action_id], device=device))
            traj.append(z.cpu().numpy())
        trajectories[ACTION_NAMES[action_id]] = np.vstack(traj)
    all_z = np.vstack(list(trajectories.values()))
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_z)
    traj_2d = {}
    idx = 0
    for name in ACTION_NAMES:
        traj_2d[name] = all_2d[idx:idx+num_steps+1]
        idx += num_steps + 1
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['gray', 'blue', 'red', 'green']
    for i, (name, traj) in enumerate(traj_2d.items()):
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], label=name, markersize=8, linewidth=2)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=200, marker='s', edgecolor='black')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=200, marker='^', edgecolor='black')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('traj')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'action_trajectories.png'), dpi=150)
    plt.close()
    print("traj ok")


def analyze_dimension_over_time(ae, dyn, states, device, num_steps=25):
    print("6 dim")
    idx0 = min(100, len(states) - 1)
    s0 = states[idx0:idx0+1].to(device)
    with torch.no_grad():
        _, z0 = ae(s0)
    z = z0.clone()
    changes = torch.zeros(LATENT_DIM)
    for _ in range(10):
        with torch.no_grad():
            z_new = dyn(z, torch.tensor([3], device=device))
        changes += (z_new - z).abs().squeeze().cpu()
        z = z_new
    top_dims = changes.argsort(descending=True)[:3].tolist()
    dim_to_track = top_dims[0]
    print("track", dim_to_track)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax_idx, action_id in enumerate(range(4)):
        ax = axes[ax_idx // 2, ax_idx % 2]
        z = z0.clone()
        values = [z[0, dim_to_track].item()]
        for _ in range(num_steps):
            with torch.no_grad():
                z = dyn(z, torch.tensor([action_id], device=device))
            values.append(z[0, dim_to_track].item())
        ax.plot(values, 'o-', linewidth=2, markersize=6, color=f'C{ax_idx}')
        ax.axhline(y=values[0], color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'z[{dim_to_track}]')
        ax.set_title(f'Action: {ACTION_NAMES[action_id]}')
        ax.grid(True, alpha=0.3)
        trend = "up" if values[-1] > values[0] else "down" if values[-1] < values[0] else "flat"
        ax.text(0.95, 0.95, f'{trend} d={values[-1]-values[0]:+.2f}', transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.suptitle(f'dim {dim_to_track}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'dim_{dim_to_track}_evolution.png'), dpi=150)
    plt.close()
    print("dim ok")
    return dim_to_track


def analyze_jump_parabola(ae, dyn, states, device, num_steps=20):
    print("6 jump")
    idx0 = min(100, len(states) - 1)
    s0 = states[idx0:idx0+1].to(device)
    with torch.no_grad():
        _, z = ae(s0)
    traj = [z.cpu().numpy()]
    for _ in range(num_steps):
        with torch.no_grad():
            z = dyn(z, torch.tensor([2], device=device))
        traj.append(z.cpu().numpy())
    traj = np.vstack(traj)
    best_dim, best_r2 = 0, 0
    t = np.arange(len(traj))
    for d in range(LATENT_DIM):
        coeffs = np.polyfit(t, traj[:, d], 2)
        fitted = np.polyval(coeffs, t)
        ss_res = np.sum((traj[:, d] - fitted) ** 2)
        ss_tot = np.sum((traj[:, d] - np.mean(traj[:, d])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 > best_r2 and coeffs[0] < 0:
            best_r2 = r2
            best_dim = d
    print("height dim", best_dim, "r2", best_r2)
    fig, ax = plt.subplots(figsize=(10, 6))
    values = traj[:, best_dim]
    ax.plot(values, 'bo-', markersize=8, linewidth=2, label='values')
    coeffs = np.polyfit(t, values, 2)
    fitted = np.polyval(coeffs, t)
    ax.plot(fitted, 'r--', linewidth=2, label=f'R2={best_r2:.3f}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'z[{best_dim}]')
    ax.set_title(f'dim {best_dim}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'jump_parabola.png'), dpi=150)
    plt.close()
    print("jump ok")
    return best_dim, best_r2


def test_stability(ae, dyn, states, device, num_steps=50):
    print("stability", num_steps)
    idx0 = min(100, len(states) - 1)
    s0 = states[idx0:idx0+1].to(device)
    with torch.no_grad():
        _, z = ae(s0)
    norms = [z.norm().item()]
    for _ in range(num_steps):
        with torch.no_grad():
            z = dyn(z, torch.tensor([3], device=device))
        norms.append(z.norm().item())
    plt.figure(figsize=(10, 4))
    plt.plot(norms, 'b-', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('||z||')
    plt.title('norm')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, 'stability.png'), dpi=150)
    plt.close()
    if max(norms) > 100 or min(norms) < 0.01:
        print("unstable")
    else:
        print("stable")
    print("stability ok")


def generate_report(dim_tracked, height_dim, height_r2):
    report = f"""1 latent {LATENT_DIM} actions 4
2 track dim {dim_tracked}
3 height dim {height_dim} r2 {height_r2:.3f}
4 done
"""
    print(report)
    with open(os.path.join(RESULT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("report", RESULT_DIR)


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

    dyn = LatentDynamicsModel(LATENT_DIM, action_dim=4).to(device)
    dyn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'dynamics.pt'), map_location=device))
    dyn.eval()

    evaluate_rollout(ae, dyn, states, actions, next_states, device, num_steps=10)
    evaluate_action_trajectories(ae, dyn, states, device, num_steps=15)
    dim_tracked = analyze_dimension_over_time(ae, dyn, states, device)
    height_dim, height_r2 = analyze_jump_parabola(ae, dyn, states, device)
    test_stability(ae, dyn, states, device)
    generate_report(dim_tracked, height_dim, height_r2)
    print("done")


if __name__ == '__main__':
    main()
