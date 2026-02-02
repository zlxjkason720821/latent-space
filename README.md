# latent-space

Mario latent-space physics: real game data only, no synthetic data. Run collect_data then train to get results.

Super-Mario-Bros.exe in root is not used. Scripts use Python gym-super-mario-bros (built-in NES). On Windows install Microsoft C++ Build Tools then pip install.

## 1 Install
cd to project root, install C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ then
py -m pip install -r requirements.txt

## 2 Collect data
py scripts/collect_data.py
or --quick for 2000 transitions
output in data/

## 3 Train
py scripts/train.py

## 4 Evaluate
py scripts/evaluate.py

## 5 Physics blending
py scripts/physics_blending.py

## 6 Play in latent (after training)
py scripts/play_in_latent.py
output: results/play_frames, results/play_in_latent_strip.png

Options:
  --data data/mario_data_xxx.npz  data file
  --steps 25                      rollout steps
  --actions "1,1,1,3,3,2,1,3"     action sequence (0=NOOP 1=RIGHT 2=JUMP 3=RIGHT+JUMP)
e.g. py scripts/play_in_latent.py --actions "1,3,1,3" --steps 30

Note: not real-time game control; rollout in latent with learned dynamics from initial state + action sequence, then decode to frames.
How we know it is jumping: evaluate.py finds a latent dimension that tracks height (parabola when applying JUMP). Each play run also saves results/play_in_latent_strip_<tag>.png and results/latent_height_<tag>.png: the height-dim curve rises when JUMP/RIGHT+JUMP is applied (orange band), then falls (parabola).

## Quick play (after train)
Double-click or run from project root:
- play_right.bat   run right  -> results/play_frames_right/, play_in_latent_strip_right.png
- play_jump.bat    run and jump -> results/play_frames_jump/, play_in_latent_strip_jump.png
- play_mixed.bat   mixed -> results/play_frames_mixed/, play_in_latent_strip_mixed.png
- play_calm.bat    slow -> results/play_frames_calm/, play_in_latent_strip_calm.png

Or: py scripts/play_presets.py [right|jump|mixed|calm|aggressive] [--steps 30]
Each run uses its own folder: results/play_frames_<tag>/ and play_in_latent_strip_<tag>.png
Manual: py scripts/play_in_latent.py --tag myrun --actions "1,3,1" --steps 20

## Run all
run_all.bat or run_all_auto.bat (Mario env must be installed)
