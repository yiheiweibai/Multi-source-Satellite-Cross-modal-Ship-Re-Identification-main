@echo off
REM ============================================================
REM 13_train_champion.bat - retrain ALL 8 members of the champion preset final_full
REM   Wave-5 full-data retrain (labels.csv = 3534 ids / 7422 images).
REM   Members (same recipe as the champion, only the train CSV changed):
REM     ship_reid (5, ship_reid_vit\venv):
REM       train_vit_full              -> outputs\full_vit              (champion epoch_100)
REM       train_convnext_t_full       -> outputs\full_convnext_t       (champion epoch_080)
REM       train_convnext_s_full       -> outputs\full_convnext_s       (champion epoch_060)
REM       train_convnext_s_pure_full  -> outputs\full_convnext_s_pure  (champion epoch_040)
REM       train_swin_s_v2_full        -> outputs\full_swin_s_v2        (champion epoch_060)
REM     SDF-Net (3, .venv-sdfnet):
REM       configs\SDF-Net-full.yml      -> logs\SDF-Net-full      (champion transformer_45 / _25)
REM       configs\SDF-Net-mos-full.yml  -> logs\SDF-Net-mos-full  (champion transformer_75)
REM
REM   Resumable: each member is skipped when its champion checkpoint already exists.
REM   ship_reid members additionally resume from outputs\<dir>\last.pth when present.
REM   Prerequisites: ship_reid_vit\venv (ship_reid_vit\01_setup.bat) and
REM                  .venv-sdfnet + SDF-Net\data\HOSS (12_sdfnet_setup.bat).
REM   Note: full-data members have seen every identity, so there is NO leak-free
REM         local validation; epoch picks follow the previous champion (exp_006).
REM ============================================================
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "REID=%ROOT%.."
set "VIT_DIR=%ROOT%ship_reid_vit"
set "SDF_DIR=%ROOT%SDF-Net"
set "VIT_PY=%VIT_DIR%\venv\Scripts\python.exe"
set "SDF_PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "DATA=%REID%\question6-data\traindata"

set "FAIL="

echo ============================================================
echo [13] Champion (final_full) member retraining
echo      root : %ROOT%
echo ============================================================

if not exist "%VIT_PY%" (
    echo [13] ERROR: ship_reid venv not found: %VIT_PY%
    echo      run ship_reid_vit\01_setup.bat first.
    pause
    exit /b 1
)
if not exist "%SDF_PY%" (
    echo [13] ERROR: SDF-Net venv not found: %SDF_PY%
    echo      run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)
if not exist "%DATA%\labels.csv" (
    echo [13] ERROR: full labels.csv not found: %DATA%\labels.csv
    pause
    exit /b 1
)

REM ============================================================
REM Part 1 - ship_reid members (5), cwd must be ship_reid_vit
REM ============================================================
echo.
echo [13] ---- Part 1/2: ship_reid members ----
cd /d "%VIT_DIR%"

call :train_vit train_vit_full              full_vit              epoch_100.pth
call :train_vit train_convnext_t_full       full_convnext_t       epoch_080.pth
call :train_vit train_convnext_s_full       full_convnext_s       epoch_060.pth
call :train_vit train_convnext_s_pure_full  full_convnext_s_pure  epoch_040.pth
call :train_vit train_swin_s_v2_full        full_swin_s_v2        epoch_060.pth

REM ============================================================
REM Part 2 - SDF-Net members (3), cwd must be SDF-Net
REM ============================================================
echo.
echo [13] ---- Part 2/2: SDF-Net members ----

if not exist "%SDF_DIR%\data\HOSS\bounding_box_train" (
    echo [13] ERROR: HOSS data not found: %SDF_DIR%\data\HOSS
    echo      run 12_sdfnet_setup.bat first.
    set "FAIL=1"
    goto :done
)

if exist "%SDF_DIR%\data\full\HOSS\bounding_box_train" (
    echo [13] full HOSS data already present, skip preparation.
) else (
    echo [13] preparing full HOSS training data ^(labels.csv, hardlink^) ...
    "%SDF_PY%" "%ROOT%scripts\sdfnet_prepare_full.py" --labels "%DATA%\labels.csv" --out_dir "%SDF_DIR%\data\full"
    if errorlevel 1 (
        echo [13] ERROR: full HOSS data preparation failed.
        set "FAIL=1"
        goto :done
    )
)

cd /d "%SDF_DIR%"
call :train_sdf SDF-Net-full     "%ROOT%logs\SDF-Net-full"      transformer_45.pth
call :train_sdf SDF-Net-mos-full "%ROOT%logs\SDF-Net-mos-full"  transformer_75.pth

:done
echo.
if defined FAIL (
    echo ============================================================
    echo [13] FINISHED WITH ERRORS - see the messages above.
    echo ============================================================
    pause
    exit /b 1
)
echo ============================================================
echo [13] DONE. All available champion member checkpoints are ready.
echo      next: 14_submit_1_env_check.bat  ^(submission chain 14-18^)
echo ============================================================
pause
endlocal
exit /b 0


REM ---------------- subroutines ----------------
:train_vit
REM %1 = config stem (ship_reid_vit\config\%1.yaml)
REM %2 = relative output dir under ship_reid_vit (used for skip / resume)
REM %3 = champion checkpoint file name under %2\checkpoints
if exist "outputs\%~2\checkpoints\%~3" (
    echo [13] skip %~1 : outputs\%~2\checkpoints\%~3 already exists
    goto :eof
)
set "RES="
if exist "outputs\%~2\last.pth" set "RES=--resume auto"
echo [13] training %~1 ...
"%VIT_PY%" train.py --config "config\%~1.yaml" !RES!
if errorlevel 1 (
    echo [13] ERROR: %~1 training failed.
    set "FAIL=1"
)
goto :eof

:train_sdf
REM %1 = config stem (SDF-Net\configs\%1.yml)
REM %2 = absolute output dir (matches OUTPUT_DIR in the yaml)
REM %3 = champion checkpoint file name under %2
if exist "%~2\%~3" (
    echo [13] skip %~1 : %~2\%~3 already exists
    goto :eof
)
echo [13] training %~1 ...
"%SDF_PY%" train.py --config_file "configs\%~1.yml"
if errorlevel 1 (
    echo [13] ERROR: %~1 training failed.
    set "FAIL=1"
)
goto :eof