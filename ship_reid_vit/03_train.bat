@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM 03_train.bat - ViT dual-branch training (Step2-4)
REM   backbone: vit_base_patch16_224 (shallow modality-specific / deep shared)
REM   losses:   SupCon + Triplet + CE + ArcFace
REM   outputs:  outputs\checkpoints\{last.pth,best.pth} + TensorBoard logs
REM Extra args are forwarded, e.g.:
REM   03_train.bat --resume auto                        (resume from outputs\checkpoints\last.pth)
REM   03_train.bat --resume outputs\checkpoints\epoch_020.pth
REM   03_train.bat --epochs 60
REM Checkpoints: last.pth / best.pth / epoch_XXX.pth (every save_every epochs, default 5)
REM ============================================================
set "PY_FILE=%CD%\venv\Scripts\python.exe"

if not exist "%PY_FILE%" (
    echo [ERROR] venv not found. Run 01_setup.bat first.
    exit /b 1
)

echo ============================================================
echo [03_train] Training with config\train_vit.yaml ...
echo ============================================================
"%PY_FILE%" train.py --config config\train_vit.yaml %*
if errorlevel 1 (
    echo [ERROR] training failed
    exit /b 1
)

echo ============================================================
echo [03_train] DONE. Next step: 04_local_val.bat
echo ============================================================
endlocal
exit /b 0