

@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "CT_IMAGE_PATH=G:\projects\ct81segapp_export\input\UZU00001_CT1_low"
set "OUTPUT_DIR=G:\projects\ct81segapp_export\output"
set N_WORKERS=8
set BATCH_SIZE=2
set DEVICE=cuda
set DICOM_NAME_REGEX=\".*\"

set DISTRO_NAME=ct81seg-ubuntu
set WSL_USER=ct81seg

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set SRC_DIR=%SCRIPT_DIR%\src
set LOG_FILE=%OUTPUT_DIR%\inference.log

mkdir "%OUTPUT_DIR%"

rem Convert PROJECT_DIR to WSL path
call :to_wsl_path %SRC_DIR%
set SRC_DIR=%WSL_PATH%
call :to_wsl_path %CT_IMAGE_PATH%
set CT_IMAGE_PATH=%WSL_PATH%
call :to_wsl_path %OUTPUT_DIR%
set OUTPUT_DIR=%WSL_PATH%

call :log "[Script start] %DATE% %TIME%"
:: Record start time
set "START_TIME=%TIME%"
wsl -d %DISTRO_NAME% -u %WSL_USER% -- bash -lc "singularity exec --nv --nvccli --bind /mnt %SRC_DIR%/resources/py3.12-torch2.8-cu12.8_latest.sif python %SRC_DIR%/main.py --image_path %CT_IMAGE_PATH% --output_dir %OUTPUT_DIR% --dicom_name_regex %DICOM_NAME_REGEX% --n_workers %N_WORKERS% --batch_size %BATCH_SIZE% --device %DEVICE%"
:: Record end time
set "END_TIME=%TIME%"
call :log "[Script end] %DATE% %TIME%"

call :time_to_cs "%START_TIME%" START_CS
call :time_to_cs "%END_TIME%"   END_CS

:: Handle midnight wrap
if %END_CS% LSS %START_CS% (
    set /a END_CS+=24*60*60*100
)

set /a ELAPSED_CS=END_CS-START_CS
set /a ELAPSED_SEC=ELAPSED_CS/100

call :log "Script elapsed: %ELAPSED_SEC% seconds"

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


:: -----------------------
:: Convert HH:MM:SS.cc -> centiseconds
:: -----------------------
:time_to_cs
setlocal EnableDelayedExpansion
set "t=%~1"
set "t=!t: =0!"
set "t=!t:,=.!"

set /a CS=(1!t:~0,2!-100)*360000 + (1!t:~3,2!-100)*6000 + (1!t:~6,2!-100)*100 + (1!t:~9,2!-100)

endlocal & set "%2=%CS%"
exit /b


:: -----------------------
:: Log and also show on screen
:: -----------------------
:log
echo %~1
>>"%LOG_FILE%" echo %~1
exit /b

REM ======================================================
REM  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
REM  Biomedical Imaging Intelligence Laboratory,
REM  Nara Institute of Science and Technology.
REM  All rights reserved.
REM  This file can not be copied and/or distributed
REM  without the express permission of Yi GU.
REM ======================================================