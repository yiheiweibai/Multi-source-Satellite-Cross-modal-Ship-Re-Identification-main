@echo off
REM setup.bat - Double-click entry for setup_env.ps1 (Ship ReID, RTX 4070 Ti SUPER / CUDA 12.4)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [setup] Python not found on PATH. Please install Python 3.8+ and check "Add Python to PATH".
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1"
if errorlevel 1 (
    echo.
    echo [setup] Setup failed. See error messages above.
    pause
    exit /b 1
)

echo.
echo [setup] Setup completed successfully.
pause
