@echo off
cd /d "%~dp0"
set "VCVARS="
set "BUILD_DIR=VC\Auxiliary\Build"
py -c "import sys; exit(0 if sys.maxsize > 2**32 else 1)" 2>nul && set "VCSCRIPT=vcvars64.bat" || set "VCSCRIPT=vcvars.bat"
echo Using C++ env script: %VCSCRIPT%

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2026\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2026\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
if exist "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2026\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\%BUILD_DIR%\%VCSCRIPT%"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\%BUILD_DIR%\%VCSCRIPT%"

if not defined VCVARS (
    for /d %%D in ("C:\Program Files (x86)\Microsoft Visual Studio\*") do (
        if exist "%%D\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=%%D\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
    )
    for /d %%D in ("C:\Program Files\Microsoft Visual Studio\*") do (
        if exist "%%D\BuildTools\%BUILD_DIR%\%VCSCRIPT%" set "VCVARS=%%D\BuildTools\%BUILD_DIR%\%VCSCRIPT%"
    )
)
if not defined VCVARS (
    echo Searching for %VCSCRIPT% under Program Files...
    for /f "delims=" %%F in ('dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\%VCSCRIPT%" 2^>nul') do (
        if not defined VCVARS set "VCVARS=%%F"
    )
    for /f "delims=" %%F in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\%VCSCRIPT%" 2^>nul') do (
        if not defined VCVARS set "VCVARS=%%F"
    )
)
if defined VCVARS (
    echo Using C++ build env.
    call "%VCVARS%"
) else (
    echo.
    echo ERROR: %VCSCRIPT% not found. Do this:
    echo 1. Open Start Menu, search "Native Tools" and run:
    echo    - "x64 Native Tools Command Prompt for VS 2026" if you use 64-bit Python
    echo    - "x86 Native Tools Command Prompt for VS 2026" if you use 32-bit Python
    echo 2. In that window run:
    echo    cd /d "%~dp0"
    echo    install_mario.bat
    echo.
)

py -m pip install -r requirements.txt
pause
