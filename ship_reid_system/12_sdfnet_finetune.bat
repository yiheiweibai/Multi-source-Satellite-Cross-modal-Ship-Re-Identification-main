@echo off
chcp 65001 >nul
REM 12_sdfnet_finetune.bat - SDF-Net 短 epoch 微调（默认 60 epoch，可配置以控制训练时间）
REM   训练：以官方预训练权重 SDF-Net.pth 为 backbone 起点（MODEL.PRETRAIN_CHOICE=imagenet，
REM         load_param 只加载 base.* 主干，身份头/解耦头随机初始化），在 HOSS 目录结构
REM         data\HOSS\bounding_box_train 上微调；训练中每 EVAL_PERIOD=5 epoch 在 HOSS 验证协议
REM         上评测，保存 best.pth 到 logs\SDF-Net-finetune\。
REM   微调后：自动用 best.pth 跑本地验证（base 与 rerank+QE 两组），与零训练初判对比。
REM 可配置（环境变量，优先于默认值）：
REM   SDF_EPOCHS        微调 epoch 数，默认 60（官方默认 100；60 起步兼顾效果与训练时间）
REM   SDF_IMS_PER_BATCH 训练 batch size，默认 32（16GB 显存建议 32，OOM 时设 16 再试）
REM   SDF_FINETUNE_DIR  输出目录，默认 logs\SDF-Net-finetune
REM 前提：
REM   1. 已运行 12_sdfnet_setup.bat（.venv-sdfnet / SDF-Net 仓库 / 官方权重 / HOSS 数据就绪）
REM   2. 本机有 NVIDIA GPU（训练依赖 CUDA；PyTorch 2.2.2+cu118）
REM 说明：训练命令等价于官方
REM       python train.py --config_file configs/SDF-Net.yml
REM   但追加命令行覆盖：PRETRAIN_PATH=官方权重、ROOT_DIR=../data、MAX_EPOCHS=60 等。
cd /d "%~dp0"

set ROOT=%~dp0
set SDF_DIR=%ROOT%SDF-Net
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth

if defined SDF_EPOCHS (set EPOCHS=%SDF_EPOCHS%) else (set EPOCHS=60)
if defined SDF_IMS_PER_BATCH (set IMS=%SDF_IMS_PER_BATCH%) else (set IMS=32)
if defined SDF_FINETUNE_DIR (set FT_DIR=%SDF_FINETUNE_DIR%) else (set FT_DIR=%ROOT%logs\SDF-Net-finetune)

REM ---- 0. 环境检查 ----
if not exist "%PY%" (
    echo [12] 未找到 .venv-sdfnet，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
if not exist "%WEIGHT%" (
    echo [12] 未找到官方权重 SDF-Net.pth，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
if not exist "%ROOT%data\HOSS\bounding_box_train" (
    echo [12] 未找到 HOSS 训练数据，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
"%PY%" -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >nul 2>nul
if errorlevel 1 (
    echo [12] 当前环境无可用 CUDA。训练必须在 NVIDIA GPU 机器上执行；
    echo     零训练初判与融合可在本机继续（12_sdfnet_zero_shot.bat / 12_sdfnet_fusion.bat）。
    pause
    exit /b 1
)

REM ---- 1. 微调训练（官方权重为起点，短 epoch 控制训练时间）----
echo.
echo [12] 开始微调：EPOCHS=%EPOCHS%  IMS_PER_BATCH=%IMS%  输出=%FT_DIR%
cd /d "%SDF_DIR%"
"%PY%" train.py --config_file configs\SDF-Net.yml ^
    MODEL.PRETRAIN_CHOICE imagenet ^
    MODEL.PRETRAIN_PATH "logs\SDF-Net\SDF-Net.pth" ^
    DATASETS.ROOT_DIR "../data" ^
    SOLVER.MAX_EPOCHS %EPOCHS% ^
    SOLVER.IMS_PER_BATCH %IMS% ^
    OUTPUT_DIR "%FT_DIR%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] 微调失败，错误码 %EC%（若为显存不足，请 set SDF_IMS_PER_BATCH=16 后重试）
    pause
    exit /b %EC%
)

REM ---- 2. 微调后本地验证：base 与 rerank+QE 两组 ----
echo.
echo [12] 微调完成，用 best.pth 跑本地验证...
set "FT_WEIGHT=%FT_DIR%\best.pth"
if not exist "%FT_WEIGHT%" (
    echo [12] 未找到 %FT_WEIGHT%，请检查训练输出目录。
    pause
    exit /b 1
)
set SIM_DIR=%ROOT%sims\sdfnet
cd /d "%SDF_DIR%"
"%PY%" ..\scripts\sdfnet_inference.py --config_file configs\SDF-Net.yml --weight "%FT_WEIGHT%" --task_json "%TASK_JSON%" --out_prediction "%SIM_DIR%\prediction_finetune_base.json"
"%PY%" ..\scripts\sdfnet_inference.py --config_file configs\SDF-Net.yml --weight "%FT_WEIGHT%" --task_json "%TASK_JSON%" --rerank --qe --out_prediction "%SIM_DIR%\prediction_finetune_rerankqe.json"

echo.
echo [12] 微调后评测（base 与 rerank+QE）：
"%PY%" ..\evaluate.py --submission --prediction "%SIM_DIR%\prediction_finetune_base.json" --task "%TASK_JSON%" --gt "%GT_JSON%"
"%PY%" ..\evaluate.py --submission --prediction "%SIM_DIR%\prediction_finetune_rerankqe.json" --task "%TASK_JSON%" --gt "%GT_JSON%"

echo.
echo [12] 微调完成。权重: %FT_WEIGHT%
echo     与 12_sdfnet_zero_shot.bat 的零训练初判结果对比，确认微调增益后决定是否启用；
echo     若要跑跨模型融合，请运行 12_sdfnet_fusion.bat（融合默认使用 base 特征 sim）。
pause
