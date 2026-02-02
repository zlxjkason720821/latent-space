"""
Play in latent space: start from real state, step with actions 0-3 (NOOP RIGHT JUMP RIGHT+JUMP),
only dynamics + decode, no real game. Saves predicted frames to results/play_frames/.
Real Mario data required: run collect_data.py first then train.py.
"""
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
import matplotlib.patches as mpatches
import glob
import argparse

from models import StateAutoencoder, LatentDynamicsModel

LATENT_DIM = 32
STATE_DIM = 169
MODEL_DIR = 'saved_models'
RESULT_DIR = 'results'
PLAY_FRAMES_DIR = 'play_frames'
ACTION_NAMES = ['NOOP', 'RIGHT', 'JUMP', 'RIGHT+JUMP']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--actions', type=str, default='1,1,1,3,3,2,1,3')
    parser.add_argument('--tag', type=str, default='', help='output subfolder name, e.g. right, jump -> results/play_frames_right/')
    args = parser.parse_args()
    out_subdir = PLAY_FRAMES_DIR + ('_' + args.tag if args.tag else '')
    strip_name = 'play_in_latent_strip' + ('_' + args.tag if args.tag else '') + '.png'

    if args.data is None:
        cands = sorted(glob.glob(os.path.join(ROOT, 'data', 'mario_data_*.npz')), key=os.path.getmtime, reverse=True)
        if not cands:
            print("no npz run collect_data then train")
            return
        args.data = cands[0]
    print("data", args.data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(args.data)
    states = torch.FloatTensor(data['states'])
    actions = [int(x) for x in args.actions.replace(' ', '').split(',')]
    actions = (actions * ((args.steps // len(actions)) + 1))[:args.steps]

    ae = StateAutoencoder(STATE_DIM, LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'autoencoder.pt'), map_location=device))
    ae.eval()
    dyn = LatentDynamicsModel(LATENT_DIM, action_dim=4).to(device)
    dyn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'dynamics.pt'), map_location=device))
    dyn.eval()

    idx0 = min(500, len(states) - 1)
    s0 = states[idx0:idx0+1].to(device)
    with torch.no_grad():
        _, z = ae(s0)

    view_size = 13
    out_dir = os.path.join(RESULT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    z_traj = [z.cpu().numpy().flatten()]
    z_cur = z
    with torch.no_grad():
        decoded_0 = ae.decode(z_cur).cpu().numpy().reshape(view_size, view_size)
        frames.append(decoded_0)
        for i, a in enumerate(actions):
            a_t = torch.tensor([a], device=device)
            z_cur = dyn(z_cur, a_t)
            z_traj.append(z_cur.cpu().numpy().flatten())
            dec = ae.decode(z_cur).cpu().numpy().reshape(view_size, view_size)
            frames.append(dec)
    z_traj = np.vstack(z_traj)

    for i, img in enumerate(frames):
        plt.imsave(os.path.join(out_dir, f'frame_{i:03d}.png'), img, cmap='gray')
    print("saved", len(frames), "frames", out_dir)

    n_show = min(15, len(frames))
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 1.2, 1.5))
    for i in range(n_show):
        axes[i].imshow(frames[i], cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title('start')
        else:
            axes[i].set_title(ACTION_NAMES[actions[i-1]])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, strip_name), dpi=120, bbox_inches='tight')
    plt.close()
    print("strip", os.path.join(RESULT_DIR, strip_name))

    height_dim = None
    report_path = os.path.join(RESULT_DIR, 'report.txt')
    if os.path.isfile(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('3 height dim'):
                    parts = line.split()
                    if len(parts) >= 4:
                        height_dim = int(parts[3])
                    break
    if height_dim is None:
        height_dim = 1
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(z_traj.shape[0])
    ax.plot(steps, z_traj[:, height_dim], 'b-o', markersize=4, label=f'z[{height_dim}] (height-like)')
    for i in range(min(len(actions), len(steps) - 1)):
        if actions[i] in (2, 3):
            ax.axvspan(steps[i], steps[i + 1], alpha=0.2, color='orange')
    ax.set_xlabel('Step')
    ax.set_ylabel(f'z[{height_dim}]')
    ax.set_title('Latent "height" dim: rises when JUMP/RIGHT+JUMP, then falls (parabola)')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor='orange', alpha=0.3, label='JUMP / RIGHT+JUMP'))
    ax.legend(handles=handles, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    height_plot = 'latent_height' + ('_' + args.tag if args.tag else '') + '.png'
    plt.savefig(os.path.join(RESULT_DIR, height_plot), dpi=120, bbox_inches='tight')
    plt.close()
    print("height curve", os.path.join(RESULT_DIR, height_plot))


if __name__ == '__main__':
    main()
