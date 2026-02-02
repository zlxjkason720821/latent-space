import torch
import torch.nn as nn
import torch.nn.functional as F


class StateAutoencoder(nn.Module):
    def __init__(self, state_dim=169, latent_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=64, residual=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        action_onehot = F.one_hot(action, self.action_dim).float()
        x = torch.cat([z, action_onehot], dim=-1)
        delta = self.net(x)
        return (z + delta) if self.residual else delta

    def rollout(self, z0, actions):
        traj = [z0.unsqueeze(0)]
        z = z0.unsqueeze(0)
        for a in actions:
            a_tensor = torch.tensor([a], device=z.device)
            z = self.forward(z, a_tensor)
            traj.append(z)
        return torch.cat(traj, dim=0)


def ae_loss(recon, original):
    return F.mse_loss(recon, original)


def dynamics_loss(z_pred, z_true):
    return F.mse_loss(z_pred, z_true)


# Possible further improvements (would need retrain):
# - AE: deeper/wider, BatchNorm, Dropout(0.05), latent_dim 64
# - Dynamics: deeper net, multi-step rollout loss, LayerNorm
# - Training: lr scheduler, separate lr for AE vs dyn

if __name__ == '__main__':
    batch, state_dim, latent_dim = 16, 169, 32
    states = torch.rand(batch, state_dim)
    actions = torch.randint(0, 4, (batch,))
    ae = StateAutoencoder(state_dim, latent_dim)
    recon, z = ae(states)
    print(f"AE: {states.shape} -> latent {z.shape} -> recon {recon.shape}")
    dyn = LatentDynamicsModel(latent_dim, action_dim=4)
    z_next = dyn(z, actions)
    print(f"Dynamics: z_next {z_next.shape}")
    traj = dyn.rollout(z[0], [1, 1, 3, 3, 2])
    print(f"Rollout: {traj.shape}")
