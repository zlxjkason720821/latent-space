@echo off
cd /d "%~dp0"
py scripts/play_in_latent.py --tag mixed --actions "1,1,1,3,3,2,1,3,1,2,3,1" --steps 30
echo Output: results\play_frames_mixed\ and results\play_in_latent_strip_mixed.png
pause
