@echo off
chcp 65001 >nul
REM 12_sdfnet_zero_shot.bat - SDF-Net 零训练初判（官方 HOSS 预训练权重直接推理评测）
REM   组合一（base）   : 官方权重 + 纯特征相似度，输出 sims\sdfnet（供融合复用）
REM   组合二（rerankqe）: 官方权重 + k-reciprocal rerank + QE（按 query_type 分组后处理，与修复后的
REM                       transoss_inference.py 同逻辑），确认跨模态分组修复对 SDF-Net 同样生效
REM   两组均基于同一份 local_val_task.json 评测（evaluate.py --submission，输出 O2S/S2O/O2O 与综合得分）
REM 前提：
REM   1. 已运行 12_sdfnet_setup.bat（.venv-sdfnet / SDF-Net 仓库 / 官方权重 / HOSS 数据就绪）
REM   2. 赛题验证集 local_val_task.json / local_val_gt.json 存在（优先复用，不重新划分）
REM 说明：零训练初判用于回答"官方权重在本地验证集上的直接表现"，据此决定是否需要微调；
REM       若在无 GPU 机器上运行，自动回退 CPU（速度慢，仅用于功能验证）。
cd /d "%~dp0"

set ROOT=%~dp0
set SDF_DIR=%ROOT%SDF-Net
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth
set SIM_DIR=%ROOT%sims\sdfnet

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
if not exist "%TASK_JSON%" (
    echo [12] 未找到 local_val_task.json，请先准备本地验证集。
    pause
    exit /b 1
)
if not exist "%GT_JSON%" (
    echo [12] 未找到 local_val_gt.json，请先准备本地验证集。
    pause
    exit /b 1
)
if not exist "%SDF_DIR%\configs\SDF-Net.yml" (
    echo [12] 未找到 SDF-Net 配置，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)

REM ---- 1. 组合一：base（纯特征，导出 sim 供融合）----
echo.
echo [12] 组合一：零训练 base（官方权重 + 纯特征）...
cd /d "%SDF_DIR%"
"%PY%" ..\scripts\sdfnet_inference.py --config_file configs\SDF-Net.yml --weight "%WEIGHT%" --task_json "%TASK_JSON%" --out_prediction "%SIM_DIR%\prediction_zero_base.json" --save_sim "%SIM_DIR%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] base 推理失败，错误码 %EC%
    pause
    exit /b %EC%
)
"%PY%" ..\evaluate.py --submission --prediction "%SIM_DIR%\prediction_zero_base.json" --task "%TASK_JSON%" --gt "%GT_JSON%"

REM ---- 2. 组合二：rerank + QE（按 query_type 分组后处理）----
echo.
echo [12] 组合二：零训练 rerank+QE（k1=20 k2=6 lambda=0.3）...
cd /d "%SDF_DIR%"
"%PY%" ..\scripts\sdfnet_inference.py --config_file configs\SDF-Net.yml --weight "%WEIGHT%" --task_json "%TASK_JSON%" --rerank --qe --out_prediction "%SIM_DIR%\prediction_zero_rerankqe.json"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] rerankqe 推理失败，错误码 %EC%
    pause
    exit /b %EC%
)
"%PY%" ..\evaluate.py --submission --prediction "%SIM_DIR%\prediction_zero_rerankqe.json" --task "%TASK_JSON%" --gt "%GT_JSON%"

echo.
echo [12] 零训练初判完成：
echo     预测文件: %SIM_DIR%\prediction_zero_base.json / prediction_zero_rerankqe.json
echo     相似度:   %SIM_DIR%\sim.pt（base，供 12_sdfnet_fusion.bat 复用）
echo     上方两行评测即为官方权重在本地验证集上的 O2S/S2O/O2O 与综合得分。
echo     若 rerankqe 综合得分未高于 base（正增益），训练/融合时建议采用 base 特征。
pause
