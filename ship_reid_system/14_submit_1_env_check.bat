@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-sdfnet\Scripts\python.exe"
set "ENV_BAT=%TEMP%\shipreid_repro_env.bat"

echo ================================================================
echo  [14] Step 1/5  Environment check
echo ================================================================
echo.

if not exist "%PY%" (
    echo [ERROR] .venv-sdfnet not found: %PY%
    echo         Run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)

REM ---- resolve configs/reproduce.yaml (pass --preset NAME to override) ----
"%PY%" "%ROOT%scripts\repro_cfg.py" dump --env-out "%ENV_BAT%" %*
if errorlevel 1 (
    echo [ERROR] failed to resolve configs/reproduce.yaml
    pause
    exit /b 1
)
call "%ENV_BAT%"

echo.
echo [14] checking inputs for preset %RRF_PRESET% ...
"%PY%" "%ROOT%scripts\repro_cfg.py" check %*
if errorlevel 1 (
    echo.
    echo [ERROR] required inputs are missing ^(see the list above^).
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  [14] DONE. Next: 15_submit_2_sdfnet_infer.bat
echo ================================================================
pause
exit /b 0