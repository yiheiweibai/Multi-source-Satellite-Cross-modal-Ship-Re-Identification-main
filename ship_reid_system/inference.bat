@echo off
chcp 65001 >nul
REM inference.bat - 生成提交用 prediction.json（TTA + 重排序 + QE）
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [inference] 未找到 .venv，请先运行 setup.bat 创建环境。
    pause
    exit /b 1
)

set CKPT=outputs/checkpoints/best.pth
if not exist "%CKPT%" (
    echo [inference] 未找到 checkpoint: %CKPT%
    echo            请先运行 train.bat 完成训练。
    pause
    exit /b 1
)

echo [inference] 正在生成 prediction.json（TTA + rerank + QE + cluster）...
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt "%CKPT%" ^
    --task_json "../赛题6-初赛/初赛测试数据/task.json" ^
    --out_prediction outputs/prediction.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 (
    echo.
    echo [inference] 推理失败，请查看上方日志。
    pause
    exit /b 1
)

echo.
echo [inference] prediction.json 已保存至: outputs/prediction.json
pause
