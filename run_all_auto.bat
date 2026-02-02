@echo off
cd /d "%~dp0"
set "PY="
if defined MARIO_PY "%MARIO_PY%" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=%MARIO_PY%"
if not defined PY py -3.13-64 -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=py -3.13-64"
if not defined PY py -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=py"
if not defined PY python -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=python"
if not defined PY if exist "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" "C:\ProgramData\miniconda3\python.exe" -c "import sys; exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1 && "C:\ProgramData\miniconda3\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\ProgramData\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "C:\Users\admin\miniconda3\python.exe" "C:\Users\admin\miniconda3\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\Users\admin\miniconda3\python.exe"
if not defined PY if exist "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not defined PY (
    echo.
    echo Mario env not found. Installing requirements for current Python...
    echo Run install_mario.bat first if nes-py fails to build.
    echo.
)
if not defined PY py -m pip install -r requirements.txt
if not defined PY py -3.13-64 -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=py -3.13-64"
if not defined PY py -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=py"
if not defined PY python -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=python"
if not defined PY if exist "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" "C:\ProgramData\miniconda3\python.exe" -c "import sys; exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1 && "C:\ProgramData\miniconda3\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=C:\ProgramData\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" -c "import sys; exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1 && "%USERPROFILE%\miniconda3\python.exe" -c "import gym_super_mario_bros" >nul 2>&1 && set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" (
    echo Miniconda is 32-bit but nes-py needs 64-bit Python.
    echo Install 64-bit Python or run from x64 Native Tools and: set MARIO_PY=path-to-64bit-python.exe
    set "PY="
)
if not defined PY (
    echo.
    echo nes-py needs 64-bit Python. Miniconda here is 32-bit.
    echo Run from "x64 Native Tools Command Prompt" and run this again.
    echo Or install 64-bit Python and set MARIO_PY=path-to-64bit-python.exe
    pause
    exit /b 1
)
echo Using: %PY%
if not exist data mkdir data
if not exist saved_models mkdir saved_models
if not exist results mkdir results

echo [1/6] collect data...
%PY% scripts/collect_data.py --quick
if errorlevel 1 (echo collect failed; pause; exit /b 1)

echo [2/6] train...
%PY% scripts/train.py
if errorlevel 1 (echo train failed; pause; exit /b 1)

echo [3/6] evaluate...
%PY% scripts/evaluate.py
if errorlevel 1 (echo evaluate failed; pause; exit /b 1)

echo [4/6] physics blending...
%PY% scripts/physics_blending.py

echo [5/6] play in latent...
%PY% scripts/play_in_latent.py

echo [6/6] done. Check results\ and saved_models\
pause
