@echo off
REM inference.bat - ?????? prediction.json??? TTA + ??? + QE?
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [inference] ??? .venv????? setup.bat ????
    pause
    exit /b 1
)

set CKPT=outputs/checkpoints/best.pth
if not exist "%CKPT%" (
    echo [inference] ??? checkpoint: %CKPT%
    echo            ???? train.bat ????
    pause
    exit /b 1
)

echo [inference] ?? prediction.json?TTA + rerank + QE + cluster?...
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt "%CKPT%" ^
    --task_json "../??6-??/??????/task.json" ^
    --out_prediction outputs/prediction.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 (
    echo.
    echo [inference] ?????????????
    pause
    exit /b 1
)

echo.
echo [inference] prediction.json ???: outputs/prediction.json
pause
