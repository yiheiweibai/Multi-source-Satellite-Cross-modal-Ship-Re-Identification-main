@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM 01_setup.bat - create venv, install dependencies, download ViT weights
REM Non-interactive. Change TORCH_INDEX below for a different CUDA build.
REM ============================================================
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "PY_FILE=%CD%\venv\Scripts\python.exe"

echo ============================================================
echo [01_setup] Working dir: %CD%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH. Install Python 3.10+ first.
    exit /b 1
)

REM ---------- 1. create venv ----------
if exist "%PY_FILE%" (
    echo [1/4] venv already exists, skip creation
) else (
    echo [1/4] Creating venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] failed to create venv
        exit /b 1
    )
)

REM ---------- 2. upgrade pip ----------
echo [2/4] Upgrading pip ...
"%PY_FILE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed
    exit /b 1
)

REM ---------- 3. install torch / torchvision ----------
echo [3/4] Installing torch / torchvision, index = %TORCH_INDEX%
"%PY_FILE%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [WARN] CUDA build failed, falling back to CPU build ...
    "%PY_FILE%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] torch install failed
        exit /b 1
    )
)

REM ---------- 4. install remaining requirements ----------
echo [4/4] Installing remaining requirements ...
"%PY_FILE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] dependency install failed
    exit /b 1
)

REM ---------- download ViT pretrained weights ----------
echo [extra] Downloading ViT pretrained weights to weights\ ...
"%PY_FILE%" scripts\download_vit_pretrained.py --name vit_base_patch16_224 --out weights\vit_base_patch16_224.pth
if errorlevel 1 (
    echo [WARN] ViT weights download failed; training will fall back to timm online download
)

echo ============================================================
echo [01_setup] DONE. Next step: 02_prepare_data.bat
echo ============================================================
endlocal
exit /b 0