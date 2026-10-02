@echo off
chcp 65001 >nul
REM run_all.bat - 一键全流程：构建本地验证 -> 训练 -> 本地验证 -> 生成提交结果
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [run_all] 未找到 .venv，请先运行 setup.bat 创建环境。
    pause
    exit /b 1
)

echo ============================================================
echo  步骤 1/4: 构建本地验证集（身份 80/20 划分）
echo ============================================================
if not exist "../赛题6-初赛/训练数据/local_val_task.json" (
    .venv\Scripts\python.exe scripts/build_local_val.py ^
        --csv "../赛题6-初赛/训练数据/labels.csv" ^
        --out_dir "../赛题6-初赛/训练数据"
    if errorlevel 1 ( pause & exit /b 1 )
) else (
    echo  本地验证任务已存在，跳过构建。
)

echo.
echo ============================================================
echo  步骤 2/4: GPU 训练
echo ============================================================
.venv\Scripts\python.exe train.py --config config/train_gpu.yaml
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  步骤 3/4: 本地验证推理（TTA + rerank + QE + cluster）并评测
echo ============================================================
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt outputs/checkpoints/best.pth ^
    --task_json "../赛题6-初赛/训练数据/local_val_task.json" ^
    --out_prediction outputs/local_val_pred.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 ( pause & exit /b 1 )

.venv\Scripts\python.exe evaluate.py --submission ^
    --prediction outputs/local_val_pred.json ^
    --task "../赛题6-初赛/训练数据/local_val_task.json" ^
    --gt "../赛题6-初赛/训练数据/local_val_gt.json"
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  步骤 4/4: 生成提交结果 prediction.json
echo ============================================================
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt outputs/checkpoints/best.pth ^
    --task_json "../赛题6-初赛/初赛测试数据/task.json" ^
    --out_prediction outputs/prediction.json ^
    --tta --rerank --qe --cluster
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  全流程完成，提交文件: outputs/prediction.json
echo ============================================================
pause
