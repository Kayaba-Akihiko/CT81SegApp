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
set "DISTRO_NAME=ct81seg-ubuntu"
rem ===================

echo Checking if %DISTRO_NAME% is installed...
wsl -d %DISTRO_NAME% -- echo OK >nul 2>&1
if not %errorlevel%==0 (
    echo %DISTRO_NAME% is NOT installed. Nothing to change
) else (
    echo %DISTRO_NAME% is installed. Uninstalling %DISTRO_NAME% ...
    wsl --unregister %DISTRO_NAME%
)
echo Done.
endlocal
pause
exit /b 0
