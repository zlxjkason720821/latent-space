# latent-space

Mario 潜空间物理学习：真实游戏数据，无合成数据。需先 collect_data 再 train 才有结果。

Super-Mario-Bros.exe 不参与运行；脚本使用 Python gym-super-mario-bros（内置 NES）。Windows 需安装 Microsoft C++ Build Tools 再 pip install。

## 1 安装
cd 到项目根目录，安装 C++ Build Tools：https://visualstudio.microsoft.com/visual-cpp-build-tools/ 然后
py -m pip install -r requirements.txt

## 2 采集数据
py scripts/collect_data.py
或 --quick 采 2000 条
输出在 data/

## 3 训练
py scripts/train.py

## 4 评估
py scripts/evaluate.py

## 5 物理混合
py scripts/physics_blending.py

## 6 潜空间“玩”（训练后）
py scripts/play_in_latent.py
输出: results/play_frames、results/play_in_latent_strip.png

可选参数:
  --data data/mario_data_xxx.npz  指定数据文件
  --steps 25                      推演步数
  --actions "1,1,1,3,3,2,1,3"     动作序列(0=NOOP 1=RIGHT 2=JUMP 3=RIGHT+JUMP)
例: py scripts/play_in_latent.py --actions "1,3,1,3" --steps 30

说明: 非实时操控真游戏，而是用学到的潜空间动力学从初始状态+动作序列推演，解码成连续帧图。

## 一键全流程
run_all.bat 或 run_all_auto.bat（需已安装 Mario 环境）
