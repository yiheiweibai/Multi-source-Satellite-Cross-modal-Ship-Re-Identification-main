@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"

echo ================================================================
echo  [18] Step 5/5  Submission verification
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

if not exist "%OUT_PRED%" (
    echo [ERROR] submission not found: %OUT_PRED%
    echo         Run 17_submit_4_rrf_fuse.bat first.
    pause
    exit /b 1
)

"%PY%" "%REPRO_DIR%\scripts\check_prediction.py" --prediction "%OUT_PRED%" --task "%TEST_TASK%" --topk %RRF_TOP_K%
if errorlevel 1 (
    echo.
    echo [ERROR] submission check FAILED. Do NOT upload this file.
    pause
    exit /b 1
)

echo.
echo [18] local-validation reference ^(1450 queries^):
echo        active preset: %RRF_PRESET%  ^(k=%RRF_K%^)  -^> Final %LOCAL_VAL_FINAL%
echo        fold0 %LOCAL_VAL_FOLD0% / fold1 %LOCAL_VAL_FOLD1%     (single-model baseline: E0 ep80 0.5888)
echo.
echo ================================================================
echo  ALL DONE. Submission file: %OUT_PRED%
echo ================================================================
pause
exit /b 0