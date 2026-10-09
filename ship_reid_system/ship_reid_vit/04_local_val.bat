@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM 04_local_val.bat - local validation (Step5: inference + postprocess + eval)
REM   1) predict on local validation protocol (TTA + k-reciprocal rerank)
REM   2) score O2S / S2O / O2O and the weighted overall score
REM Prerequisite: 02_prepare_data.bat and 03_train.bat have been run.
REM Local val paths come from config\train_vit.yaml, no non-ASCII args here.
REM ============================================================
set "PY_FILE=%CD%\venv\Scripts\python.exe"
set "CKPT=outputs\checkpoints\best.pth"
set "PRED=outputs\local_val_prediction.json"

if not exist "%PY_FILE%" (
    echo [ERROR] venv not found. Run 01_setup.bat first.
    exit /b 1
)
if not exist "%CKPT%" (
    echo [ERROR] checkpoint %CKPT% not found. Run 03_train.bat first.
    exit /b 1
)

echo ============================================================
echo [04_local_val] Generating local validation predictions ...
echo ============================================================
"%PY_FILE%" inference.py --config config\train_vit.yaml --ckpt "%CKPT%" ^
    --local_val_task --out_prediction "%PRED%" --tta --rerank
if errorlevel 1 (
    echo [ERROR] local validation inference failed
    exit /b 1
)

echo ============================================================
echo [04_local_val] Scoring with competition metrics ...
echo ============================================================
"%PY_FILE%" evaluate.py --submission --config config\train_vit.yaml ^
    --local_val --prediction "%PRED%"
if errorlevel 1 (
    echo [ERROR] evaluation failed
    exit /b 1
)

echo ============================================================
echo [04_local_val] DONE. Next step: 05_inference.bat
echo ============================================================
endlocal
exit /b 0