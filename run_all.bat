@echo off
cd /d "%~dp0"
set PY=
where py >nul 2>&1 && set PY=py
if "%PY%"=="" set PY=python

echo 0 dirs
if not exist data mkdir data
if not exist saved_models mkdir saved_models
if not exist results mkdir results

echo 1 deps
%PY% -c "import torch; import numpy; print('ok')" 2>nul || (
    %PY% -m pip install torch numpy matplotlib tqdm scikit-learn scipy
)
%PY% -c "import gym_super_mario_bros" 2>nul || (
    echo no mario env install requirements.txt first then run again
    pause
    exit /b 1
)

echo 2 data
%PY% scripts/collect_data.py
if errorlevel 1 (echo fail; pause; exit /b 1)

echo 3 train
%PY% scripts/train.py
if errorlevel 1 (pause; exit /b 1)

echo 4 eval
%PY% scripts/evaluate.py
if errorlevel 1 (pause; exit /b 1)

echo 5 blend
%PY% scripts/physics_blending.py

echo 6 play latent
%PY% scripts/play_in_latent.py

echo done results\ saved_models\
pause
