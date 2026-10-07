@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM 05_inference.bat - test-set inference, produce submission prediction.json
REM   Parses the official task.json (queries + gallery), filters candidates by
REM   query_type, returns top-10 gallery image_id per query, then self-checks.
REM Extra args are forwarded, e.g.:  05_inference.bat --qe --cluster
REM The test task path comes from config\train_vit.yaml, no non-ASCII args here.
REM ============================================================
set "PY_FILE=%CD%\venv\Scripts\python.exe"
set "CKPT=outputs\checkpoints\best.pth"
set "PRED=outputs\prediction.json"

if not exist "%PY_FILE%" (
    echo [ERROR] venv not found. Run 01_setup.bat first.
    exit /b 1
)
if not exist "%CKPT%" (
    echo [ERROR] checkpoint %CKPT% not found. Run 03_train.bat first.
    exit /b 1
)

echo ============================================================
echo [05_inference] Running test-set inference ...
echo ============================================================
"%PY_FILE%" inference.py --config config\train_vit.yaml --ckpt "%CKPT%" ^
    --use_test_task --out_prediction "%PRED%" --tta --rerank %*
if errorlevel 1 (
    echo [ERROR] test-set inference failed
    exit /b 1
)

echo ============================================================
echo [05_inference] DONE. Submission file: %PRED%
echo ============================================================
endlocal
exit /b 0