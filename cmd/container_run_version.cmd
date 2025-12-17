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

set DISTRO_NAME=ct81seg-ubuntu
set WSL_USER=ct81seg
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set SRC_DIR=%SCRIPT_DIR%\src

:: Convert Windows DIR to WSL path
call :to_wsl_path %SRC_DIR%
set SRC_DIR=%WSL_PATH%

echo Checking version ...
wsl -d %DISTRO_NAME% -u %WSL_USER% -- bash -lc "singularity exec --bind /mnt %SRC_DIR%/resources/py3.12-torch2.8-cu12.8_latest.sif python %SRC_DIR%/main.py --version"

endlocal
pause
exit /b 0

:: -----------------------
:: Function definitions
:: -----------------------
:: Convert a Windows path like G:\projects\ct_81_seg_export
:: to a WSL path like /mnt/g/projects/ct_81_seg_export
:to_wsl_path
setlocal EnableDelayedExpansion

rem Original Windows path
set "WIN=%~1"

rem Replace backslashes with slashes: G:\foo\bar -> G:/foo/bar
set "TMP=%WIN:\=/%"

rem Get drive letter (G)
set "DRIVE=!TMP:~0,1!"

rem Force drive letter to lowercase
for %%L in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /I "!DRIVE!"=="%%L" set "DRIVE=%%L"
)

rem Strip "G:" -> "/projects/ct_81_seg_export"
set "PATH_NO_DRIVE=!TMP:~2!"

rem Build /mnt/g/projects/ct_81_seg_export
set "OUT=/mnt/!DRIVE!!PATH_NO_DRIVE!"

endlocal & set "WSL_PATH=%OUT%"
exit /b