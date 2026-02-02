@echo off
cd /d "%~dp0"
where cl >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: C++ compiler "cl.exe" is not in PATH.
    echo You must run this script FROM the Developer Command Prompt.
    echo.
    echo Steps:
    echo 1. Press Win, search:  Native Tools
    echo 2. Open:  "x64 Native Tools Command Prompt for VS 2026"
    echo    - If you use 32-bit Python, open "x86 Native Tools ..." instead
    echo 3. In that window run:
    echo    cd /d "%~dp0"
    echo    install_in_dev_prompt.bat
    echo.
    pause
    exit /b 1
)
echo C++ compiler found. Installing Python packages...
py -m pip install -r requirements.txt
echo.
pause
