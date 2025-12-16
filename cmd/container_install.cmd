@echo off
REM ======================================================
REM  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
REM  Biomedical Imaging Intelligence Laboratory,
REM  Nara Institute of Science and Technology.
REM  All rights reserved.
REM  This file can not be copied and/or distributed
REM  without the express permission of Yi GU.
REM ======================================================
setlocal

rem ===== CONFIG =====
set DISTRO_NAME=ct81seg-ubuntu
set WSL_USER=ct81seg
rem ===================

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set SRC_DIR=%SCRIPT_DIR%\src

echo Installing WSL platform (if needed)...
wsl --install --no-distribution >nul 2>&1

echo Checking if %DISTRO_NAME% is installed...
wsl -d %DISTRO_NAME% -- echo OK >nul 2>&1
if not %errorlevel%==0 (
    echo %DISTRO_NAME% is NOT installed.
    echo Installing %DISTRO_NAME% to "%LOCALAPPDATA%\wsl\%DISTRO_NAME%" ...
    mkdir "%LOCALAPPDATA%\wsl"
    wsl --import "%DISTRO_NAME%" "%LOCALAPPDATA%\wsl\%DISTRO_NAME%" "%SRC_DIR%\resources\%DISTRO_NAME%.tar"
    rem Optional: wait a bit
    timeout /t 5 /nobreak >nul
) else (
    echo %DISTRO_NAME% is installed. Skipping installation.
)

echo Verifying %DISTRO_NAME% ...
wsl -d %DISTRO_NAME% -- echo OK >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [ERROR] %DISTRO_NAME% cannot be started.
    echo   - If this is the first time of WSL installation, please REBOOT the machine and rerun this script again.
    pause
    goto :EOF
)


echo Done.
endlocal
pause
exit /b 0