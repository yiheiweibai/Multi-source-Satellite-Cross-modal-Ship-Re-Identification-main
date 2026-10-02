@echo off
chcp 65001 >nul
REM local_val.bat - 本地验证：用训练集划分验证集评测并输出 GT 指标
REM 前置：先运行 scripts/build_local_val.py 生成 local_val_task.json + local_val_gt.json
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [local_val] 未找到 .venv，请先运行 setup.bat 创建环境。
    pause
    exit /b 1
)

set CKPT=outputs/checkpoints/best.pth
if not exist "%CKPT%" (
    echo [local_val] 未找到 checkpoint: %CKPT%
    pause
    exit /b 1
)

set TASK=../赛题6-初赛/训练数据/local_val_task.json
set GT=../赛题6-初赛/训练数据/local_val_gt.json

if not exist "%TASK%" (
    echo [local_val] 未找到验证任务: %TASK%
    echo            请先运行: python scripts/build_local_val.py
    pause
    exit /b 1
)

echo [local_val] 正在推理（TTA + rerank + QE + cluster）...
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt "%CKPT%" ^
    --task_json "%TASK%" ^
    --out_prediction outputs/local_val_pred.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 (
    echo.
    echo [local_val] 推理失败
    pause
    exit /b 1
)

echo.
echo [local_val] 开始评测...
.venv\Scripts\python.exe evaluate.py --submission ^
    --prediction outputs/local_val_pred.json ^
    --task "%TASK%" ^
    --gt "%GT%"
if errorlevel 1 (
    echo.
    echo [local_val] 评测失败
    pause
    exit /b 1
)

echo.
echo [local_val] 完成
pause
