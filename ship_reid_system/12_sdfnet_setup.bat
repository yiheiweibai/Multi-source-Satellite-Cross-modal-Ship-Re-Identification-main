@echo off
chcp 65001 >nul
REM 12_sdfnet_setup.bat - SDF-Net 环境准备（一键：依赖 / 官方权重 / HOSS 数据转换）
REM   1. 创建独立虚拟环境 .venv-sdfnet（Python 3.9+，与 TransOSS 的 .venv 隔离，避免 timm 版本冲突）
REM   2. 安装 PyTorch 2.2.2+cu118 与 SDF-Net 其余依赖（timm / yacs / opencv-python / Pillow 等）
REM   3. 克隆官方仓库 https://github.com/cfrfree/SDF-Net 到 SDF-Net\（已存在则跳过）
REM   4. 下载官方预训练权重 SDF-Net.pth（HuggingFace: Chenfree233/SDF-Net，已存在则跳过）
REM   5. 将赛题训练数据转换为 HOSS 目录结构 data\HOSS\{bounding_box_train,bounding_box_test,query}
REM      优先复用已存在的 local_val_task.json（不重新划分），保证与 09/11 结果横向可比
REM 说明：
REM   - SDF-Net 要求 Python 3.9+ / PyTorch 2.2.2+cu118；本脚本面向 16GB 显存机器
REM     （SDF-Net.yml 默认 IMS_PER_BATCH=32 在该显存下可训练，若 OOM 请用
REM       set SDF_IMS_PER_BATCH=16 后再运行 12_sdfnet_finetune.bat）
REM   - 若本机无 NVIDIA GPU，cu118 安装会失败，可用 CPU 版 torch 做零训练推理初判
REM     （速度较慢），训练请在 GPU 机器上执行
cd /d "%~dp0"

set ROOT=%~dp0
set SDF_DIR=%ROOT%SDF-Net
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth

echo [12] SDF-Net 环境准备开始...

REM ---- 1. 检查系统 Python 版本 ----
where python >nul 2>nul
if errorlevel 1 (
    echo [12] 未找到 python，请先安装 Python 3.9+ 并加入 PATH。
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [12] 系统 Python 版本: %PYVER%

REM ---- 2. 创建虚拟环境（已存在则跳过）----
if exist "%PY%" (
    echo [12] 虚拟环境已存在: .venv-sdfnet
) else (
    echo [12] 创建虚拟环境 .venv-sdfnet ...
    python -m venv "%ROOT%.venv-sdfnet"
    if errorlevel 1 (
        echo [12] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

REM ---- 3. 安装依赖：先 cu118 版 torch/torchvision，再安装其余依赖 ----
"%PY%" -m pip install --upgrade pip >nul
echo [12] 安装 PyTorch 2.2.2+cu118 / torchvision 0.17.2+cu118 ...
"%PY%" -m pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [12] cu118 安装失败。若本机无 NVIDIA GPU，可改用 CPU 版仅做零训练推理：
    echo     "%PY%" -m pip install torch==2.2.2 torchvision==0.17.2
    echo     然后重新运行本脚本；训练仍需在 GPU 机器完成。
    pause
    exit /b 1
)
echo [12] 安装 SDF-Net 其余依赖（timm/yacs/opencv-python/Pillow 等）...
if exist "%SDF_DIR%\requirements.txt" (
    findstr /V /C:"torch" "%SDF_DIR%\requirements.txt" > "%ROOT%temp\sdfnet_req_no_torch.txt"
    "%PY%" -m pip install -r "%ROOT%temp\sdfnet_req_no_torch.txt"
    if errorlevel 1 (
        echo [12] 依赖安装失败，请查看上方错误。
        pause
        exit /b 1
    )
)

REM ---- 4. 克隆 SDF-Net 仓库（已存在则跳过）----
if exist "%SDF_DIR%\.git" (
    echo [12] SDF-Net 仓库已存在，跳过克隆。
) else (
    echo [12] 克隆 https://github.com/cfrfree/SDF-Net ...
    git clone https://github.com/cfrfree/SDF-Net "%SDF_DIR%"
    if errorlevel 1 (
        echo [12] 克隆失败，请检查网络。
        pause
        exit /b 1
    )
)

REM ---- 5. 下载官方预训练权重（已存在则跳过）----
if exist "%WEIGHT%" (
    echo [12] 官方权重已存在: %WEIGHT%
) else (
    echo [12] 下载官方预训练权重 SDF-Net.pth（HuggingFace: Chenfree233/SDF-Net，约 333MB）...
    "%PY%" scripts\sdfnet_download_weights.py --out "%WEIGHT%"
    if errorlevel 1 (
        echo [12] 权重下载失败。可手动下载后放置到:
        echo     %WEIGHT%
        pause
        exit /b 1
    )
)

REM ---- 6. 数据转换：赛题数据 -^> HOSS 目录结构（存在则跳过）----
if exist "%ROOT%data\HOSS\bounding_box_train" (
    echo [12] HOSS 数据已存在，跳过转换: data\HOSS\{bounding_box_train,bounding_box_test,query}
) else (
    if not exist "%TASK_JSON%" (
        echo [12] 未找到 local_val_task.json，请先运行 09_local_val_transoss.bat 或手动生成验证集。
        pause
        exit /b 1
    )
    if not exist "%DATA_DIR%\labels_train.csv" (
        echo [12] 未找到 labels_train.csv，请先运行 09_local_val_transoss.bat 生成训练子集。
        pause
        exit /b 1
    )
    echo [12] 转换赛题数据为 HOSS 结构（复用 local_val_task.json，不重新划分）...
    "%PY%" scripts\sdfnet_prepare_data.py --labels_train "%DATA_DIR%\labels_train.csv" --labels_full "%DATA_DIR%\labels.csv" --task "%TASK_JSON%" --out_dir "%ROOT%data\HOSS"
    if errorlevel 1 (
        echo [12] 数据转换失败，请查看上方错误。
        pause
        exit /b 1
    )
)

echo.
echo [12] SDF-Net 环境准备完成。下一步：
echo     1) 零训练初判（官方权重直接推理评测）:  运行 12_sdfnet_zero_shot.bat
echo     2) 短 epoch 微调（默认 60 epoch）:      运行 12_sdfnet_finetune.bat
echo     3) 跨模型融合（SDF-Net + TransOSS）:    运行 12_sdfnet_fusion.bat
pause
