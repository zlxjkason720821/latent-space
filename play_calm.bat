@echo off
cd /d "%~dp0"
py scripts/play_in_latent.py --tag calm --actions "0,1,0,1,1,0,1,1,1,0" --steps 25
echo Output: results\play_frames_calm\ and results\play_in_latent_strip_calm.png
pause
