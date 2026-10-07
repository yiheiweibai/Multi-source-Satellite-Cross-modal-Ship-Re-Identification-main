@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM 02_prepare_data.bat - compute modality mean/std + build local val protocol
REM   1) modality-independent normalization stats (Step1)
REM   2) identity-isolated 80/20 split -> labels_train.csv
REM      local_val_task.json / local_val_gt.json
REM Paths come from config\default.yaml (data_root), no non-ASCII args here.
REM ============================================================
set "PY_FILE=%CD%\venv\Scripts\python.exe"

if not exist "%PY_FILE%" (
    echo [ERROR] venv not found. Run 01_setup.bat first.
    exit /b 1
)

echo ============================================================
echo [02_prepare_data] Computing modality mean / std ...
echo ============================================================
"%PY_FILE%" scripts\compute_stats.py --config config\default.yaml
if errorlevel 1 (
    echo [WARN] stats failed; keeping ImageNet defaults
)

echo ============================================================
echo [02_prepare_data] Building local validation protocol (80/20) ...
echo ============================================================
"%PY_FILE%" scripts\build_local_val.py --config config\default.yaml --val_ratio 0.2 --seed 42
if errorlevel 1 (
    echo [ERROR] local val protocol build failed
    exit /b 1
)

echo ============================================================
echo [02_prepare_data] DONE. Next step: 03_train.bat
echo ============================================================
endlocal
exit /b 0