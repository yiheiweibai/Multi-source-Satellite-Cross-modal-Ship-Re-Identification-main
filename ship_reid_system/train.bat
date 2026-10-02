@echo off
chcp 65001 >nul
REM train.bat - GPU 训练入口（RTX 4070 Ti SUPER / CUDA 12.4）
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [train] 未找到 .venv，请先运行 setup.bat 创建环境。
    pause
    exit /b 1
)

echo [train] 使用 config/train_gpu.yaml 开始训练...
.venv\Scripts\python.exe train.py --config config/train_gpu.yaml
if errorlevel 1 (
    echo.
    echo [train] 训练失败，请查看上方日志。
    pause
    exit /b 1
)

echo.
echo [train] 训练完成，checkpoint 已保存至 outputs/checkpoints/
pause
