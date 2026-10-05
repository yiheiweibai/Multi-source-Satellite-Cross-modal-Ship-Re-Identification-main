@echo off
chcp 65001 >nul
REM 13_sdfnet_plan.bat - SDF-Net × TransOSS 统一方案入口（一键按阶段对比 9 个组合）
REM   阶段组合（全部复用固定 local_val_task.json，不重新划分；微调仅执行一次，其余共享权重）：
REM     A1 : SDF-Net 官方预训练权重 base
REM     A2 : SDF-Net 官方预训练权重 + rerankqe（按 query_type 分组后处理）
REM     B1 : SDF-Net 微调后 base（默认 60 epoch，SDF_EPOCHS 可配）
REM     B2 : SDF-Net 微调后 + rerankqe
REM     C  : TransOSS 现有权重 + 修复后 rerankqe 重测
REM     D1~D3 : SDF-Net(微调后) × TransOSS 分数融合 weighted，SDF 权重 0.5/0.6/0.7
REM     D4 : 同上 RRF 融合
REM   完成后自动汇总生成对比文档：
REM     experiments\exp_005_sdfnet_plan_compare.md
REM   包含完整结果表、每组合 Δ vs TransOSS baseline（0.4831/0.4881）、
REM   正增益标记（>0.005 才算）与推荐提交组合。
REM 可配置（环境变量，优先于默认值）：
REM   SDF_EPOCHS        微调 epoch 数，默认 60
REM   SDF_IMS_PER_BATCH 微调 batch size，默认 32（16GB 显存建议 32，OOM 时设 16）
REM   SDF_FINETUNE_DIR  微调输出目录，默认 logs\SDF-Net-finetune
REM   SDF_WEIGHT        官方权重路径，默认 SDF-Net\logs\SDF-Net\SDF-Net.pth
REM   TRANS_WEIGHT      TransOSS 权重路径，默认 Hoss-ReID\logs\competition_transoss\transformer_200.pth
REM 前提：
REM   1. SDF-Net 侧已运行 12_sdfnet_setup.bat（.venv-sdfnet / SDF-Net 仓库 / 官方权重 / HOSS 数据）
REM   2. TransOSS 侧已运行 08_transoss_setup.bat（.venv / Hoss-ReID / transformer_200.pth）——缺失时 C/D 自动跳过
REM   3. 验证集 local_val_task.json / local_val_gt.json 存在
REM 说明：本脚本是 12_sdfnet_zero_shot / 12_sdfnet_finetune / 12_sdfnet_fusion 的统一收敛入口；
REM       12_sdfnet_setup.bat 仍保留为一次性环境准备。
cd /d "%~dp0"

set ROOT=%~dp0
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json
set SDF_DIR=%ROOT%SDF-Net
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth

if defined SDF_EPOCHS (set EPOCHS=%SDF_EPOCHS%) else (set EPOCHS=60)
if defined SDF_IMS_PER_BATCH (set IMS=%SDF_IMS_PER_BATCH%) else (set IMS=32)
if defined SDF_FINETUNE_DIR (set FT_DIR=%SDF_FINETUNE_DIR%) else (set FT_DIR=%ROOT%logs\SDF-Net-finetune)
if defined SDF_WEIGHT (set W_SDF=%SDF_WEIGHT%) else (set W_SDF=%WEIGHT%)
if defined TRANS_WEIGHT (set W_TRANS=%TRANS_WEIGHT%) else (set W_TRANS=%ROOT%Hoss-ReID\logs\competition_transoss\transformer_200.pth)

REM ---- 0. 环境检查 ----
if not exist "%PY%" (
    echo [13] 未找到 .venv-sdfnet，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
if not exist "%W_SDF%" (
    echo [13] 未找到 SDF-Net 官方权重: %W_SDF%
    echo     请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
if not exist "%TASK_JSON%" (
    echo [13] 未找到 local_val_task.json，请先准备本地验证集。
    pause
    exit /b 1
)
if not exist "%GT_JSON%" (
    echo [13] 未找到 local_val_gt.json，请先准备本地验证集。
    pause
    exit /b 1
)
if not exist "%SDF_DIR%\configs\SDF-Net.yml" (
    echo [13] 未找到 SDF-Net 配置，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)

echo.
echo [13] 统一方案入口启动：
echo      微调 epoch=%EPOCHS%   batch=%IMS%   微调输出=%FT_DIR%
echo      官方权重=%W_SDF%
echo      TransOSS 权重=%W_TRANS%
echo      （无 GPU 或权重缺失时，相关组合会自动跳过并在日志中说明）
echo.

REM ---- 1. 调用编排脚本（阶段自动推进 + 评测 + 生成对比文档）----
"%PY%" scripts\sdfnet_plan.py --epochs %EPOCHS% --ims_per_batch %IMS% --ft_dir "%FT_DIR%" --sdf_official_weight "%W_SDF%" --trans_weight "%W_TRANS%" --task_json "%TASK_JSON%" --gt_json "%GT_JSON%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [13] 方案编排失败，错误码 %EC%（详情见 sims\plan\run.log）
    pause
    exit /b %EC%
)

echo.
echo [13] 统一方案完成。输出：
echo     对比文档: %ROOT%..\experiments\exp_005_sdfnet_plan_compare.md
echo     中间产物: %ROOT%sims\plan\（pred_*.json / sim.pt / run.log）
echo     推荐提交组合请见对比文档「增益分析与推荐组合」一节。
pause
