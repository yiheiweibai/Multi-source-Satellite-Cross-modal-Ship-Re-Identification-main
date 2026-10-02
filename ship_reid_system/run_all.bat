@echo off
REM run_all.bat - ????????????? -> ?? -> ???? -> ????
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [run_all] ??? .venv????? setup.bat ????
    pause
    exit /b 1
)

echo ============================================================
echo  ?? 1/4: ???????
echo ============================================================
if not exist "../??6-??/????/local_val_task.json" (
    .venv\Scripts\python.exe scripts/build_local_val.py ^
        --csv "../??6-??/????/labels.csv" ^
        --out_dir "../??6-??/????"
    if errorlevel 1 ( pause & exit /b 1 )
) else (
    echo  ???????????
)

echo.
echo ============================================================
echo  ?? 2/4: GPU ??
echo ============================================================
.venv\Scripts\python.exe train.py --config config/train_gpu.yaml
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  ?? 3/4: ????????TTA + rerank + QE?
echo ============================================================
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt outputs/checkpoints/best.pth ^
    --task_json "../??6-??/????/local_val_task.json" ^
    --out_prediction outputs/local_val_pred.json ^
    --tta --rerank --qe
if errorlevel 1 ( pause & exit /b 1 )

.venv\Scripts\python.exe evaluate.py --submission ^
    --prediction outputs/local_val_pred.json ^
    --task "../??6-??/????/local_val_task.json" ^
    --gt "../??6-??/????/local_val_gt.json"
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  ?? 4/4: ?????? prediction.json
echo ============================================================
.venv\Scripts\python.exe inference.py --config config/train_gpu.yaml ^
    --ckpt outputs/checkpoints/best.pth ^
    --task_json "../??6-??/??????/task.json" ^
    --out_prediction outputs/prediction.json ^
    --tta --rerank --qe
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo ============================================================
echo  ?????prediction.json: outputs/prediction.json
echo ============================================================
pause
