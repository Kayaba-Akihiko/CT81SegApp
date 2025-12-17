@echo off
REM ======================================================
REM  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
REM  Biomedical Imaging Intelligence Laboratory,
REM  Nara Institute of Science and Technology.
REM  All rights reserved.
REM  This file can not be copied and/or distributed
REM  without the express permission of Yi GU.
REM ======================================================

setlocal EnableExtensions EnableDelayedExpansion

rem ===== CONFIG =====
set "DISTRO_NAME=ct81seg-ubuntu"
set "WSL_USER=ct81seg"
rem ===================

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "SRC_DIR=%SCRIPT_DIR%\src"

echo.
echo ===== Enabling required Windows features for WSL =====
dism /online /get-featureinfo /featurename:VirtualMachinePlatform | findstr /I "Enabled" >nul
if not %errorlevel%==0 (
    echo Enabling Virtual Machine Platform...
    dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    set NEED_REBOOT=1
)

if defined NEED_REBOOT (
    echo.
    echo [IMPORTANT] Windows features were enabled.
    echo Please REBOOT the machine and re-run this script.
    pause
    exit /b 0
)

echo.
echo ===== Installing WSL platform (if needed) =====
wsl --install --no-distribution >nul 2>&1

echo.
echo ===== Checking if %DISTRO_NAME% is installed =====
wsl -d %DISTRO_NAME% -- echo OK >nul 2>&1
if not %errorlevel%==0 (
    echo %DISTRO_NAME% is NOT installed.
    echo Installing %DISTRO_NAME% to "%LOCALAPPDATA%\wsl\%DISTRO_NAME%" ...

    if not exist "%LOCALAPPDATA%\wsl" mkdir "%LOCALAPPDATA%\wsl"

    wsl --import "%DISTRO_NAME%" ^
        "%LOCALAPPDATA%\wsl\%DISTRO_NAME%" ^
        "%SRC_DIR%\resources\%DISTRO_NAME%.tar"

    timeout /t 5 /nobreak >nul
) else (
    echo %DISTRO_NAME% is already installed. Skipping.
)

echo.
echo ===== Verifying %DISTRO_NAME% =====
wsl -d %DISTRO_NAME% -- echo OK >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [ERROR] %DISTRO_NAME% cannot be started.
    echo   - Pleas restart and try again.
    pause
    exit /b 1
)

echo.
echo ===== WSL setup completed successfully =====
endlocal
pause
exit /b 0