@echo off
chcp 65001 >nul
REM 05_pseudo_label.bat - 伪标签自训练：对测试集 gallery 打伪标签 -> 合并训练 -> 重新训练
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [pseudo] 未找到 .venv，请先运行 01_setup.bat 创建环境。
    pause
    exit /b 1
)

set CKPT=outputs/checkpoints/best.pth
if not exist "%CKPT%" (
    echo [pseudo] 未找到 checkpoint: %CKPT%
    pause
    exit /b 1
)

echo [pseudo] 第 1 步：为测试集 gallery 打伪标签（阈值 0.85）...
.venv\Scripts\python.exe scripts/pseudo_label.py ^
    --ckpt "%CKPT%" ^
    --train_csv "../赛题6-初赛/训练数据/labels_train.csv" ^
    --task_json "../赛题6-初赛/初赛测试数据/task.json" ^
    --out_csv "../赛题6-初赛/训练数据/labels_pseudo.csv" ^
    --threshold 0.85
if errorlevel 1 (
    echo.
    echo [pseudo] 伪标签生成失败
    pause
    exit /b 1
)

echo.
echo [pseudo] 第 2 步：使用合并后的伪标签重新训练...
.venv\Scripts\python.exe train.py --config config/train_gpu.yaml ^
    train_labels_csv="../赛题6-初赛/训练数据/labels_pseudo.csv"
if errorlevel 1 (
    echo.
    echo [pseudo] 重新训练失败
    pause
    exit /b 1
)

echo.
echo [pseudo] 伪标签自训练完成
pause
