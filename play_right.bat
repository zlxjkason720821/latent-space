@echo off
cd /d "%~dp0"
py scripts/play_in_latent.py --tag right --actions "1,1,1,1,1,1,1,1,1,1" --steps 30
echo Output: results\play_frames_right\ and results\play_in_latent_strip_right.png
pause
