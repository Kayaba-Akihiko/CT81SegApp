

@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CT_IMAGE_PATH=G:\projects\ct81segapp_export\input\UZU00001_CT1_low
set OUTPUT_DIR=G:\projects\ct_81_seg_v2_export\output
set N_WORKERS=8
set BATCH_SIZE=2
set DEVICE=cuda
set DICOM_NAME_REGEX=\".*\"

set DISTRO_NAME=ct81seg-ubuntu
set WSL_USER=ct81seg

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set SRC_DIR=%SCRIPT_DIR%\src
set LOG_DIR=%OUTPUT_DIR%\inference.log

mkdir "%OUTPUT_DIR%"

rem Convert PROJECT_DIR to WSL path
call :to_wsl_path %SRC_DIR%
set SRC_DIR=%WSL_PATH%
call :to_wsl_path %CT_IMAGE_PATH%
set CT_IMAGE_PATH=%WSL_PATH%
call :to_wsl_path %OUTPUT_DIR%
set OUTPUT_DIR=%WSL_PATH%

echo CT_IMAGE_PATH=%CT_IMAGE_PATH%
echo OUTPUT_DIR=%OUTPUT_DIR%

echo [Script start] >> %LOG_PATH%
REM --- start time ---
for /f "tokens=1-4 delims=:.," %%a in ("%TIME%") do (
    set /a "START_MS=(((1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100))*1000 + (1%%d-100))"
)

wsl -d %DISTRO_NAME% -u %WSL_USER% -- bash -lc "singularity exec --nv --nvccli --bind /mnt %SRC_DIR%/py3.12-torch2.8-cu12.8_latest.sif python %SRC_DIR%/main.py --image_path %CT_IMAGE_PATH% --output_dir %OUTPUT_DIR% --dicom_name_regex %DICOM_NAME_REGEX% --n_workers %N_WORKERS% --batch_size %BATCH_SIZE% --device %DEVICE%"

REM --- end time ---
for /f "tokens=1-4 delims=:.," %%a in ("%TIME%") do (
    set /a "END_MS=(((1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100))*1000 + (1%%d-100))"
)
echo [Script end] >> %LOG_PATH%

REM --- handle midnight wrap ---
if !END_MS! LSS !START_MS! (
    set /a END_MS+=24*3600*1000
)

set /a ELAPSED_MS=END_MS-START_MS
set /a ELAPSED_SEC=ELAPSED_MS/1000

echo Script elapsed: !ELAPSED_SEC! seconds >> %LOG_PATH%

endlocal
pause
goto :eof   rem <-- make sure we don't "fall into" the function below


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

REM ======================================================
REM  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
REM  Biomedical Imaging Intelligence Laboratory,
REM  Nara Institute of Science and Technology.
REM  All rights reserved.
REM  This file can not be copied and/or distributed
REM  without the express permission of Yi GU.
REM ======================================================