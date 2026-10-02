@echo off
REM train.bat - GPU ???RTX 4070 Ti SUPER / CUDA 12.4?
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [train] ??? .venv????? setup.bat ????
    pause
    exit /b 1
)

echo [train] ?? config/train_gpu.yaml ??...
.venv\Scripts\python.exe train.py --config config/train_gpu.yaml
if errorlevel 1 (
    echo.
    echo [train] ?????????????
    pause
    exit /b 1
)

echo.
echo [train] ?????checkpoint ??? outputs/checkpoints/
pause
