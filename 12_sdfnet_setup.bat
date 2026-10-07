@echo off
REM 12_sdfnet_setup.bat - SDF-Net environment preparation (venv / deps / official weights / HOSS data)
REM   1. Create an isolated venv .venv-sdfnet (separate from the TransOSS .venv to avoid timm conflicts)
REM   2. Install PyTorch 2.2.2+cu118, torchvision 0.17.2+cu118 and the remaining SDF-Net dependencies
REM   3. Clone https://github.com/cfrfree/SDF-Net into SDF-Net\ (skip if the folder already exists)
REM   4. Download the official weights SDF-Net.pth (skip if it already exists)
REM      Note: the official checkpoint is trained at 256x128, so its pos_embed is
REM      [1,130,768] and it CANNOT be used directly with 256x256 input.
REM   5. Derive the 256x256 variant SDF-Net_256.pth (skip if it already exists)
REM      via scripts\sdfnet_resize_posembed.py (pos_embed bilinear 256x128 -> 256x256)
REM   6. Convert the competition data into the HOSS layout:
REM        SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}
REM      The existing local_val_task.json is reused (no re-split) so results stay comparable.
REM Notes:
REM   - Data directory can be overridden with the SDF_DATA_DIR environment variable.
REM   - SDF-Net.yml default IMS_PER_BATCH=32 fits a 16GB GPU. If you get OOM,
REM     lower IMS_PER_BATCH in the yaml (19_mos_retrain.bat trains with the same yaml).
cd /d "%~dp0"

set ROOT=%~dp0
set SDF_DIR=%ROOT%SDF-Net
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
if defined SDF_DATA_DIR (set DATA_DIR=%SDF_DATA_DIR%) else (set DATA_DIR=h:\Ship-Re-Identification\question6-data\traindata)
set TASK_JSON=%DATA_DIR%\local_val_task.json
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth
set WEIGHT_256=%SDF_DIR%\logs\SDF-Net\SDF-Net_256.pth
set HOSS_DIR=%SDF_DIR%\data\HOSS

echo [12] SDF-Net setup start...

REM ---- 1. check system python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [12] python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [12] system python: %PYVER%

REM ---- 2. create venv (skip if present) ----
if exist "%PY%" (
    echo [12] venv already exists: .venv-sdfnet
) else (
    echo [12] creating venv .venv-sdfnet ...
    python -m venv "%ROOT%.venv-sdfnet"
    if errorlevel 1 (
        echo [12] failed to create venv.
        pause
        exit /b 1
    )
)

REM ---- 3. install dependencies ----
"%PY%" -m pip install --upgrade pip >nul
echo [12] installing torch 2.2.2+cu118 / torchvision 0.17.2+cu118 ...
"%PY%" -m pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [12] cu118 install failed. On a machine without an NVIDIA GPU use the CPU build instead:
    echo     "%PY%" -m pip install torch==2.2.2 torchvision==0.17.2
    echo     then re-run this script. Training still requires a GPU machine.
    pause
    exit /b 1
)
echo [12] installing remaining dependencies (numpy / opencv-python / Pillow / thop / timm / yacs) ...
"%PY%" -m pip install numpy==1.25.0 opencv-python==4.11.0.86 Pillow==11.3.0 thop==0.1.1.post2209072238 timm==1.0.25 yacs==0.1.8
if errorlevel 1 (
    echo [12] dependency install failed, see the errors above.
    pause
    exit /b 1
)

REM ---- 4. clone the SDF-Net repository (skip if present) ----
if exist "%SDF_DIR%\train.py" (
    echo [12] SDF-Net repo already present, skip clone.
) else (
    echo [12] cloning https://github.com/cfrfree/SDF-Net ...
    git clone https://github.com/cfrfree/SDF-Net "%SDF_DIR%"
    if errorlevel 1 (
        echo [12] clone failed, please check the network.
        pause
        exit /b 1
    )
)

REM ---- 5. download the official weights (skip if present) ----
if exist "%WEIGHT%" (
    echo [12] official weights already present: %WEIGHT%
) else (
    echo [12] downloading official weights SDF-Net.pth (about 333MB) ...
    "%PY%" scripts\sdfnet_download_weights.py --out "%WEIGHT%"
    if errorlevel 1 (
        echo [12] weight download failed. Place the file manually at:
        echo     %WEIGHT%
        pause
        exit /b 1
    )
)

REM ---- 6. derive the 256x256 variant (official weights are trained at 256x128) ----
if exist "%WEIGHT_256%" (
    echo [12] 256x256 weights already present: %WEIGHT_256%
) else (
    echo [12] resizing pos_embed from 256x128 to 256x256 ...
    "%PY%" scripts\sdfnet_resize_posembed.py --in_weight "%WEIGHT%" --out_weight "%WEIGHT_256%"
    if errorlevel 1 (
        echo [12] pos_embed resize failed. The final flow requires this file:
        echo     %WEIGHT_256%
        pause
        exit /b 1
    )
)

REM ---- 7. convert competition data into the HOSS layout (skip if present) ----
if exist "%HOSS_DIR%\bounding_box_train" (
    echo [12] HOSS data already present, skip conversion: %HOSS_DIR%
) else (
    if not exist "%TASK_JSON%" (
        echo [12] local_val_task.json not found: %TASK_JSON%
        pause
        exit /b 1
    )
    if not exist "%DATA_DIR%\labels_train.csv" (
        echo [12] labels_train.csv not found: %DATA_DIR%\labels_train.csv
        pause
        exit /b 1
    )
    echo [12] converting competition data into the HOSS layout (reusing local_val_task.json) ...
    "%PY%" scripts\sdfnet_prepare_data.py --labels_train "%DATA_DIR%\labels_train.csv" --labels_full "%DATA_DIR%\labels.csv" --task "%TASK_JSON%" --out_dir "%HOSS_DIR%"
    if errorlevel 1 (
        echo [12] data conversion failed, see the errors above.
        pause
        exit /b 1
    )
)

echo.
echo [12] SDF-Net setup done.
echo      HOSS data:  %HOSS_DIR%
echo      weights:    %WEIGHT%
echo      256x256:    %WEIGHT_256%
echo      next step:  run 14_submit_1_env_check.bat
pause