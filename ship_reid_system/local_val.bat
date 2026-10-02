@echo off
REM local_val.bat - ??????????? GT ??
REM ??????? scripts/build_local_val.py ?? local_val_task.json + local_val_gt.json
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [local_val] ??? .venv????? setup.bat ????
    pause
    exit /b 1
)

set CKPT=outputs/checkpoints/best.pth
if not exist "%CKPT%" (
    echo [local_val] ??? checkpoint: %CKPT%
    pause
    exit /b 1
)

set TASK=../??6-??/????/local_val_task.json
set GT=../??6-??/????/local_val_gt.json

if not exist "%TASK%" (
    echo [local_val] ??? %TASK%
    echo            ????: python scripts/build_local_val.py
    pause
    exit /b 1
)

echo [local_val] ????????TTA + rerank + QE + cluster?...
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt "%CKPT%" ^
    --task_json "%TASK%" ^
    --out_prediction outputs/local_val_pred.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 (
    echo.
    echo [local_val] ????
    pause
    exit /b 1
)

echo.
echo [local_val] ???...
.venv\Scripts\python.exe evaluate.py --submission ^
    --prediction outputs/local_val_pred.json ^
    --task "%TASK%" ^
    --gt "%GT%"
if errorlevel 1 (
    echo.
    echo [local_val] ????
    pause
    exit /b 1
)

echo.
echo [local_val] ??
pause
