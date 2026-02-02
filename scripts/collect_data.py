import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

CUSTOM_ACTIONS = [
    ['NOOP'],
    ['right'],
    ['A'],
    ['right', 'A'],
]

NUM_TRANSITIONS = 25000
MAX_STEPS_PER_EP = 400
VIEW_SIZE = 13
TILE_SIZE = 16
SAVE_DIR = 'data'


def extract_state(frame, info):
    gray = np.mean(frame, axis=2)
    mario_x = info.get('x_pos', 40)
    mario_y = info.get('y_pos', 79)
    tile_x = int(mario_x // TILE_SIZE)
    tile_y = int(mario_y // TILE_SIZE)
    half = VIEW_SIZE // 2
    tiles = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px = (tile_x + dx) * TILE_SIZE
            py = (tile_y + dy) * TILE_SIZE
            if 0 <= px < 256 - TILE_SIZE and 0 <= py < 240 - TILE_SIZE:
                region = gray[py:py+TILE_SIZE, px:px+TILE_SIZE]
                tiles[dy + half, dx + half] = np.mean(region) / 255.0
    return tiles.flatten()


def collect_data(quick=False):
    os.chdir(ROOT)
    n = 2000 if quick else NUM_TRANSITIONS
    print("collect", n)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, CUSTOM_ACTIONS)

    states, actions, next_states = [], [], []
    pbar = tqdm(total=n, desc="collect")

    while len(states) < n:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        info = result[1] if isinstance(result, tuple) and len(result) > 1 else {}

        s_t = extract_state(obs, info)

        for step in range(MAX_STEPS_PER_EP):
            if len(states) >= n:
                break
            action = np.random.choice(4, p=[0.1, 0.3, 0.2, 0.4])
            result = env.step(action)
            obs_next = result[0]
            done = result[2]
            info = result[4] if len(result) > 4 else result[3] if len(result) > 3 else {}
            if isinstance(done, (list, tuple)):
                done = done[0] if len(done) else False
            s_t1 = extract_state(obs_next, info)
            states.append(s_t)
            actions.append(action)
            next_states.append(s_t1)
            pbar.update(1)
            s_t = s_t1
            if done:
                break

    pbar.close()
    env.close()

    states = np.array(states[:n], dtype=np.float32)
    actions = np.array(actions[:n], dtype=np.int64)
    next_states = np.array(next_states[:n], dtype=np.float32)

    os.chdir(ROOT)
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(SAVE_DIR, f'mario_data_{timestamp}.npz')
    np.savez(path, states=states, actions=actions, next_states=next_states)

    print("saved", path, len(states))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    collect_data(quick=args.quick)
