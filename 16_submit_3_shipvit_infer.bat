@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"
set "VIT_STEPS_TXT=%TEMP%\shipreid_repro_env_vitsteps.txt"

echo ================================================================
echo  [16] Step 3/5  ship_reid test-set inference (TTA + rerank)
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

echo [16] preset      : %RRF_PRESET%
echo [16] member top-K: %RRF_MEMBER_TOPK%
echo [16] to run      : %VIT_N% ship_reid member(s) in this preset

if "%VIT_N%"=="0" (
    echo.
    echo ================================================================
    echo  [16] DONE ^(this preset uses no ship_reid member^).
    echo       Next: 17_submit_4_rrf_fuse.bat
    echo ================================================================
    pause
    exit /b 0
)

if not exist "%VIT_STEPS_TXT%" (
    echo [ERROR] vit steps file not found: %VIT_STEPS_TXT%
    pause
    exit /b 1
)

REM Running inference rewrites an input member of the ensemble, so it is
REM skipped when the prediction already exists. Force with:
REM     16_submit_3_shipvit_infer.bat force
set "FORCE=0"
if /i "%~1"=="force" set "FORCE=1"

cd /d "%VIT_DIR%"

for /f "usebackq tokens=1,2,3,4 delims=|" %%a in ("%VIT_STEPS_TXT%") do (
    call :one "%%a" "%%b" "%%c" "%%d"
    if errorlevel 1 exit /b 1
)

echo.
echo ================================================================
echo  [16] DONE. Next: 17_submit_4_rrf_fuse.bat
echo ================================================================
pause
exit /b 0

REM ------------------------------------------------------------------
REM :one  name  config  ckpt  pred
REM   config/ckpt come from the member's vit_config / vit_ckpt in
REM   configs/reproduce.yaml (both relative to the ship_reid_vit dir).
REM   config must be given per member: split_idx is not part of the
REM   checkpoint config backfill whitelist.
REM ------------------------------------------------------------------
:one
set "M_NAME=%~1"
set "M_CFG=%~2"
set "M_CKPT=%~3"
set "M_PRED=%~4"

if "%FORCE%"=="0" if exist "%M_PRED%" (
    echo [16] reuse existing: %M_PRED%
    exit /b 0
)

if not exist "%M_CKPT%" (
    echo [ERROR] ship_reid checkpoint not found: %M_CKPT%
    echo         Train this backbone first ^(ship_reid_vit\train.py^).
    pause
    exit /b 1
)
if not exist "%M_CFG%" (
    echo [ERROR] ship_reid config not found: %M_CFG%
    pause
    exit /b 1
)

echo.
echo [16] === %M_NAME% ===
echo [16] config: %M_CFG%
echo [16] ckpt  : %M_CKPT%
"%VIT_PY%" inference.py --config "%M_CFG%" --ckpt "%M_CKPT%" --use_test_task --out_prediction "%M_PRED%" --topk %RRF_MEMBER_TOPK% --tta --rerank
if errorlevel 1 (
    echo [ERROR] ship_reid inference failed: %M_NAME%
    pause
    exit /b 1
)
echo [16] ok ^-^> %M_PRED%
exit /b 0