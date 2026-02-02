"""
Run play_in_latent with preset action sequences. Usage:
  py scripts/play_presets.py [preset] [--steps N]
Presets: right, jump, mixed, calm, aggressive
"""
import sys
import os
import subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

PRESETS = {
    'right':    '1,1,1,1,1,1,1,1,1,1',
    'jump':     '3,3,1,3,2,3,1,3,3,1',
    'mixed':    '1,1,1,3,3,2,1,3,1,2,3,1',
    'calm':     '0,1,0,1,1,0,1,1,1,0',
    'aggressive': '3,3,3,1,3,2,3,3,1,3',
}

def main():
    preset = 'mixed'
    steps = 30
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if args:
        preset = args[0].lower()
    if '--steps' in sys.argv:
        i = sys.argv.index('--steps')
        if i + 1 < len(sys.argv):
            steps = int(sys.argv[i + 1])
    if preset not in PRESETS:
        print("Presets:", ", ".join(PRESETS))
        return
    actions = PRESETS[preset]
    script = os.path.join(ROOT, 'scripts', 'play_in_latent.py')
    subprocess.run([sys.executable, script, '--tag', preset, '--actions', actions, '--steps', str(steps)], cwd=ROOT)

if __name__ == '__main__':
    main()
