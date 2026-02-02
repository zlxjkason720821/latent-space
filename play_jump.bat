@echo off
cd /d "%~dp0"
py scripts/play_in_latent.py --tag jump --actions "3,3,1,3,2,3,1,3,3,1" --steps 30
echo Output: results\play_frames_jump\ and results\play_in_latent_strip_jump.png
pause
