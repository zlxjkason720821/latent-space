latent space mario
real Mario only. no synthetic data. results only after collect_data then train.

Super-Mario-Bros.exe in root is not used. scripts use Python gym-super-mario-bros (built-in NES). win: install Microsoft C++ Build Tools then pip install.

1 install
cd project root
install C++ Build Tools https://visualstudio.microsoft.com/visual-cpp-build-tools/ then
py -m pip install -r requirements.txt

2 data
py scripts/collect_data.py
or --quick for 2000
output in data/

3 train
py scripts/train.py

4 eval
py scripts/evaluate.py

5 blend
py scripts/physics_blending.py

6 play in latent (训练后“玩”的方式)
py scripts/play_in_latent.py
out: results/play_frames results/play_in_latent_strip.png

可选参数:
  --data data/mario_data_xxx.npz  指定数据文件
  --steps 25                      推演步数
  --actions "1,1,1,3,3,2,1,3"     动作序列(0=NOOP 1=RIGHT 2=JUMP 3=RIGHT+JUMP)
例: py scripts/play_in_latent.py --actions "1,3,1,3" --steps 30

说明: 不是实时操控真游戏，而是用学到的潜空间动力学从初始状态+动作序列推演，解码成连续帧图。

run all
run_all.bat
needs mario env. collect_data then train eval blend play_in_latent
