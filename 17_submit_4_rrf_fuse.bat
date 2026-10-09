@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"

echo ================================================================
echo  [17] Step 4/5  RRF fusion -^> prediction.json
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

echo [17] ensemble recipe from configs/reproduce.yaml:
echo        preset = %RRF_PRESET%
echo        k      = %RRF_K%
echo        topk   = %RRF_TOP_K%
echo        recipe = %RRF_MEMBERS%
echo        out    = %OUT_PRED%
echo.
echo      Override at run time, e.g.:
echo        17_submit_4_rrf_fuse.bat --preset anchor_plus_sv
echo        17_submit_4_rrf_fuse.bat --weights 1 1 1 1 1 0.5 0.75 1
echo        ^(weight count must match the active preset: final=8 members^)
echo.

REM members/weights stay inside Python (cmd would re-parse embedded quotes)
"%PY%" "%REPRO_DIR%\scripts\repro_cfg.py" fuse %*
if errorlevel 1 (
    echo.
    echo [ERROR] RRF fusion failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  [17] DONE. Next: 18_submit_5_verify.bat
echo ================================================================
pause
exit /b 0