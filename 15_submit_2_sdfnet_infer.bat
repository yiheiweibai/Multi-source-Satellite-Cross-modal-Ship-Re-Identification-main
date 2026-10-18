@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"
set "STEPS_TXT=%TEMP%\shipreid_repro_env_steps.txt"

echo ================================================================
echo  [15] Step 2/5  SDF-Net test-set inference
echo ================================================================
echo.

if not exist "%PY%" (
    echo [ERROR] .venv-sdfnet python not found: %PY%
    echo         Run 12_sdfnet_setup.bat first. If .venv-sdfnet exists but has no
    echo         Scripts folder, delete it and re-run 12_sdfnet_setup.bat.
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

if not exist "%TEST_TASK%" (
    echo [ERROR] test task.json not found: %TEST_TASK%
    pause
    exit /b 1
)
if not exist "%STEPS_TXT%" (
    echo [ERROR] steps file not found: %STEPS_TXT%
    pause
    exit /b 1
)

echo [15] preset    : %RRF_PRESET%
echo [15] test task : %TEST_TASK%
echo [15] output dir: %SDF_SIM_DIR%
echo [15] member top-K: %RRF_MEMBER_TOPK%
echo [15] to run    : %SDF_N% member(s) need local inference

if "%SDF_N%"=="0" (
    echo.
    echo ================================================================
    echo  [15] DONE ^(nothing to infer for this preset^).
    echo       Next: 16_submit_3_shipvit_infer.bat
    echo ================================================================
    pause
    exit /b 0
)

if not exist "%SDF_SIM_DIR%" mkdir "%SDF_SIM_DIR%"

REM SDF-Net.yml PRETRAIN_PATH is relative to the SDF-Net dir, so run from there
cd /d "%SDFNET_DIR%"

for /f "usebackq tokens=1,2,3 delims=|" %%a in ("%STEPS_TXT%") do (
    call :one "%%a" "%%b" "%%c"
    if errorlevel 1 exit /b 1
)

echo.
echo ================================================================
echo  [15] DONE. Next: 16_submit_3_shipvit_infer.bat
echo ================================================================
pause
exit /b 0

REM ------------------------------------------------------------------
REM :one  name  ckpt  pred
REM   Members whose checkpoint is not on this machine (e.g. sdfnet_base75,
REM   whose weights live on the teammate's D: drive) are skipped when their
REM   prediction file already exists; otherwise it is a hard error.
REM ------------------------------------------------------------------
:one
set "M_NAME=%~1"
set "M_CKPT=%~2"
set "M_PRED=%~3"

if not exist "%M_CKPT%" (
    if exist "%M_PRED%" (
        echo [15] skip %M_NAME% : ckpt not present locally, reusing existing prediction
        echo [15]   ckpt ^(missing^): %M_CKPT%
        echo [15]   pred ^(kept^)   : %M_PRED%
        exit /b 0
    )
    echo [ERROR] SDF-Net ckpt not found: %M_CKPT%
    echo         and no existing prediction at: %M_PRED%
    pause
    exit /b 1
)

echo.
echo [15] === %M_NAME% ===
"%PY%" "%REPRO_DIR%\scripts\sdfnet_inference.py" --config_file "%SDFNET_CONFIG%" --weight "%M_CKPT%" --task_json "%TEST_TASK%" --out_prediction "%M_PRED%" --topk %RRF_MEMBER_TOPK% --batch_size 16
if errorlevel 1 (
    echo [ERROR] inference failed: %M_NAME%
    pause
    exit /b 1
)
echo [15] ok ^-^> %M_PRED%
exit /b 0