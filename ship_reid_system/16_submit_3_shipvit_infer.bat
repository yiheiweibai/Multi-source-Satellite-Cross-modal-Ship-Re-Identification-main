@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"

echo ================================================================
echo  [16] Step 3/5  ship_reid_vit test-set inference (TTA + rerank)
echo ================================================================
echo.

if not exist "%PY%" (
    echo [ERROR] .venv-sdfnet not found: %PY%
    pause
    exit /b 1
)

REM ---- re-resolve the config so this step never uses a stale env file ----
"%PY%" "%ROOT%scripts\repro_cfg.py" dump --env-out "%ENV_BAT%" %*
if errorlevel 1 (
    echo [ERROR] failed to resolve configs/reproduce.yaml
    pause
    exit /b 1
)
call "%ENV_BAT%"

echo [16] preset: %RRF_PRESET%

if "%SHIPVIT_NEEDED%"=="0" (
    echo [16] this preset does not use ship_reid_vit, skip.
    echo.
    echo ================================================================
    echo  [16] DONE ^(skipped^). Next: 17_submit_4_rrf_fuse.bat
    echo ================================================================
    pause
    exit /b 0
)

if /i "%~1"=="force" goto :run_inference

if exist "%SHIPVIT_PRED%" (
    echo [16] prediction already exists, REUSED: %SHIPVIT_PRED%
    echo.
    echo      This file is an input member of the ensemble. Re-running the
    echo      inference is disabled by default so results stay comparable.
    echo      To force a re-run:  16_submit_3_shipvit_infer.bat force
    echo.
    echo ================================================================
    echo  [16] DONE ^(reused^). Next: 17_submit_4_rrf_fuse.bat
    echo ================================================================
    pause
    exit /b 0
)

:run_inference
if not exist "%SHIPVIT_PY%" (
    echo [ERROR] ship_reid_vit venv not found: %SHIPVIT_PY%
    echo         Run ship_reid_vit\01_setup.bat first.
    pause
    exit /b 1
)
if not exist "%SHIPVIT_CKPT%" (
    echo [ERROR] checkpoint not found: %SHIPVIT_CKPT%
    echo         Run ship_reid_vit\03_train.bat first.
    pause
    exit /b 1
)

cd /d "%SHIPVIT_DIR%"
echo [16] running inference ...
"%SHIPVIT_PY%" inference.py --config config\train_vit.yaml --ckpt "%SHIPVIT_CKPT%" --use_test_task --out_prediction "%SHIPVIT_PRED%" --tta --rerank
if errorlevel 1 (
    echo [ERROR] ship_reid_vit inference failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  [16] DONE. Next: 17_submit_4_rrf_fuse.bat
echo ================================================================
pause
exit /b 0