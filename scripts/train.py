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
import argparse
from tqdm import tqdm
import glob

from models import StateAutoencoder, LatentDynamicsModel, ae_loss, dynamics_loss

LATENT_DIM = 32
BATCH_SIZE = 128
AE_EPOCHS = 30
DYN_EPOCHS = 50
LR = 1e-3
SAVE_DIR = 'saved_models'


def train_autoencoder(states, device):
    print("3 ae", states.shape[1], LATENT_DIM)
    state_dim = states.shape[1]
    model = StateAutoencoder(state_dim, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(states), batch_size=BATCH_SIZE, shuffle=True)
    history = []
    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"AE Epoch {epoch+1}/{AE_EPOCHS}", leave=False):
            x = batch[0].to(device)
            recon, z = model(x)
            loss = ae_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(epoch+1, avg_loss)
    return model, history


def train_dynamics(ae, states, actions, next_states, device):
    print("4 dyn")
    ae.eval()
    z_list, z_next_list = [], []
    with torch.no_grad():
        for i in range(0, len(states), 1024):
            s = states[i:i+1024].to(device)
            s_next = next_states[i:i+1024].to(device)
            _, z = ae(s)
            _, z_next = ae(s_next)
            z_list.append(z.cpu())
            z_next_list.append(z_next.cpu())
    z_all = torch.cat(z_list)
    z_next_all = torch.cat(z_next_list)
    print("encoded", z_all.shape)

    dyn = LatentDynamicsModel(LATENT_DIM, action_dim=4).to(device)
    optimizer = optim.Adam(dyn.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(z_all, actions, z_next_all),
        batch_size=BATCH_SIZE, shuffle=True
    )
    history = []
    for epoch in range(DYN_EPOCHS):
        dyn.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Dyn Epoch {epoch+1}/{DYN_EPOCHS}", leave=False):
            z, a, z_next = [x.to(device) for x in batch]
            z_pred = dyn(z, a)
            loss = dynamics_loss(z_pred, z_next)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(epoch+1, avg_loss)
    return dyn, history, z_all, z_next_all


def plot_and_save(ae_hist, dyn_hist):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ae_hist, 'b-', linewidth=2)
    ax1.set_title('Autoencoder Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)
    ax2.plot(dyn_hist, 'r-', linewidth=2)
    ax2.set_title('Dynamics Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print("curves", SAVE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    if args.data is None:
        candidates = sorted(glob.glob(os.path.join(ROOT, 'data', 'mario_data_*.npz')), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("no npz run collect_data first")
            return
        args.data = candidates[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device", device)

    data = np.load(args.data)
    states = torch.FloatTensor(data['states'])
    actions = torch.LongTensor(data['actions'])
    next_states = torch.FloatTensor(data['next_states'])
    print(states.shape, actions.shape)

    os.makedirs(SAVE_DIR, exist_ok=True)

    ae, ae_hist = train_autoencoder(states, device)
    torch.save(ae.state_dict(), os.path.join(SAVE_DIR, 'autoencoder.pt'))
    print("ae saved")

    dyn, dyn_hist, z_all, z_next_all = train_dynamics(ae, states, actions, next_states, device)
    torch.save(dyn.state_dict(), os.path.join(SAVE_DIR, 'dynamics.pt'))
    print("dyn saved")

    torch.save({
        'z_all': z_all,
        'z_next_all': z_next_all,
        'actions': actions,
    }, os.path.join(SAVE_DIR, 'latent_data.pt'))

    plot_and_save(ae_hist, dyn_hist)
    print("done")


if __name__ == '__main__':
    main()
