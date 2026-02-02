@echo off
cd /d "%~dp0"
echo Reinstalling nes-py for 64-bit Python. Run this from "x64 Native Tools Command Prompt".
echo.
py -3.13-64 -m pip uninstall nes-py -y
py -3.13-64 -m pip install nes-py==8.2.1
echo.
echo Test: py -3.13-64 -c "import gym_super_mario_bros; print('ok')"
py -3.13-64 -c "import gym_super_mario_bros; print('ok')"
if errorlevel 1 (echo Failed. Keep using x64 Native Tools and run install_mario.bat first.) else (echo Success. Now run run_all_auto.bat)
pause
