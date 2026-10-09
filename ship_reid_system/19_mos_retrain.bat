@echo off
REM 19_mos_retrain.bat - MOS CMAL alignment-loss retraining (SDF-Net)
REM
REM Recipe: identical to the E0 run (configs/SDF-Net.yml) except
REM   MODEL.CMAL_LOSS_WEIGHT = 1.0  ->  see configs/SDF-Net-mos.yml
REM Training starts from the official SDF-Net_256 weight (PRETRAIN_CHOICE=clip),
REM NOT a resume from a mid-training checkpoint: train.py re-opens a fresh cosine
REM cycle on resume and re-heats the converged LR (documented project lesson).
REM
REM Output dir is OUTPUT_DIR inside configs/SDF-Net-mos.yml (logs/SDF-Net-mos).
REM One checkpoint every 5 epochs; select the best by Final Score afterwards:
REM   python scripts\sdfnet_ckpt_sweep.py --ckpt_dir logs\SDF-Net-mos --tag mos
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "SDF_DIR=%ROOT%SDF-Net"
set "MOS_LOG=%ROOT%logs\SDF-Net-mos"

if not exist "%PY%" (
    echo [19] .venv-sdfnet not found: %PY%
    echo      please run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)
if not exist "%SDF_DIR%\configs\SDF-Net-mos.yml" (
    echo [19] config not found: %SDF_DIR%\configs\SDF-Net-mos.yml
    pause
    exit /b 1
)

echo.
echo [19] MOS CMAL retrain start
echo      config : %SDF_DIR%\configs\SDF-Net-mos.yml
echo      output : %MOS_LOG%
echo.

pushd "%SDF_DIR%"
"%PY%" train.py --config_file configs/SDF-Net-mos.yml
set "EC=%ERRORLEVEL%"
popd

if not "%EC%"=="0" (
    echo [19] training failed with exit code %EC%
    pause
    exit /b %EC%
)

echo.
echo [19] training finished. checkpoints in: %MOS_LOG%
echo      next: python scripts\sdfnet_ckpt_sweep.py --ckpt_dir logs\SDF-Net-mos --tag mos
pause