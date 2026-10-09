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
    echo.
    echo [15] === %%a ===
    "%PY%" "%REPRO_DIR%\scripts\sdfnet_inference.py" --config_file "%SDFNET_CONFIG%" --weight "%%b" --task_json "%TEST_TASK%" --out_prediction "%%c" --topk %RRF_MEMBER_TOPK% --batch_size 16
    if errorlevel 1 (
        echo [ERROR] inference failed: %%a
        pause
        exit /b 1
    )
    echo [15] ok ^-^> %%c
)

echo.
echo ================================================================
echo  [15] DONE. Next: 16_submit_3_shipvit_infer.bat
echo ================================================================
pause
exit /b 0