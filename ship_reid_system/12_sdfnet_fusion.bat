@echo off
chcp 65001 >nul
REM 12_sdfnet_fusion.bat - SDF-Net + TransOSS 跨模型分数融合（加权平均 + RRF 两种，可配置权重）
REM   流程：
REM     1. SDF-Net 官方/微调权重跑 base 推理并导出 sim.pt/meta.json（sims\sdfnet）
REM     2. TransOSS 权重跑 base 推理并导出 sim.pt/meta.json（sims\transoss）
REM        （均使用 base 纯特征，不做 rerank/QE，保证融合输入是原始相似度）
REM     3. 调用 fuse_sdfnet_transoss.py：对齐两侧相似度矩阵，分别用
REM        weighted（min-max 归一化 + 加权平均，默认权重 SDF:TransOSS=0.6:0.4）
REM        与 rrf（倒数排名融合，默认 k=60）生成融合 prediction，并调用 evaluate.py 评测
REM     4. 汇总单模型与两种融合的 O2S/S2O/O2O 与综合得分对比
REM 可配置（环境变量）：
REM   SDF_WEIGHT           微调权重路径，默认官方权重 logs\SDF-Net\SDF-Net.pth
REM   TRANS_WEIGHT         TransOSS 权重路径，默认 logs\competition_transoss\transformer_200.pth
REM   FUSE_W_A             加权融合中 SDF-Net 权重（0~1），默认 0.6；TransOSS = 1 - FUSE_W_A
REM   FUSE_METHOD          both / weighted / rrf，默认 both
REM 前提：
REM   1. SDF-Net 侧：已运行 12_sdfnet_setup.bat（+ 可选 12_sdfnet_finetune.bat）
REM   2. TransOSS 侧：已运行 08_transoss_setup.bat 与 07_transoss_pipeline.bat
REM      （Hoss-ReID 仓库 + transformer_200.pth 权重）
REM   3. 验证集 local_val_task.json / local_val_gt.json 存在
cd /d "%~dp0"

set ROOT=%~dp0
set SDF_DIR=%ROOT%SDF-Net
set TRANS_DIR=%ROOT%Hoss-ReID
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
set PY_TRANS=%ROOT%.venv\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json
set SDF_SIM=%ROOT%sims\sdfnet
set TRANS_SIM=%ROOT%sims\transoss
set OUT_DIR=%ROOT%sims\fused

if defined SDF_WEIGHT (set W_SDF=%SDF_WEIGHT%) else (set W_SDF=%SDF_DIR%\logs\SDF-Net\SDF-Net.pth)
if defined TRANS_WEIGHT (set W_TRANS=%TRANS_WEIGHT%) else (set W_TRANS=%TRANS_DIR%\logs\competition_transoss\transformer_200.pth)
if defined FUSE_W_A (set W_A=%FUSE_W_A%) else (set W_A=0.6)
if defined FUSE_METHOD (set METHOD=%FUSE_METHOD%) else (set METHOD=both)

REM ---- 0. 环境检查 ----
if not exist "%PY%" (
    echo [12] 未找到 .venv-sdfnet，请先运行 12_sdfnet_setup.bat。
    pause
    exit /b 1
)
if not exist "%W_SDF%" (
    echo [12] 未找到 SDF-Net 权重: %W_SDF%
    pause
    exit /b 1
)
if not exist "%PY_TRANS%" (
    echo [12] 未找到 TransOSS 虚拟环境 .venv，请先运行 08_transoss_setup.bat。
    pause
    exit /b 1
)
if not exist "%TRANS_DIR%\transoss_inference.py" (
    echo [12] 未找到 Hoss-ReID 仓库，请先运行 08_transoss_setup.bat。
    pause
    exit /b 1
)
if not exist "%W_TRANS%" (
    echo [12] 未找到 TransOSS 权重: %W_TRANS%
    echo     请先运行 07_transoss_pipeline.bat 完成训练。
    pause
    exit /b 1
)
if not exist "%TASK_JSON%" (
    echo [12] 未找到 local_val_task.json，请先准备本地验证集。
    pause
    exit /b 1
)

REM ---- 1. SDF-Net 侧：base 特征 + 导出 sim ----
echo.
echo [12] 步骤 1/3：SDF-Net base 推理 + 导出相似度（%W_SDF%）...
cd /d "%SDF_DIR%"
"%PY%" ..\scripts\sdfnet_inference.py --config_file configs\SDF-Net.yml --weight "%W_SDF%" --task_json "%TASK_JSON%" --out_prediction "%SDF_SIM%\prediction_base.json" --save_sim "%SDF_SIM%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] SDF-Net 推理失败，错误码 %EC%
    pause
    exit /b %EC%
)

REM ---- 2. TransOSS 侧：base 特征 + 导出 sim ----
echo.
echo [12] 步骤 2/3：TransOSS base 推理 + 导出相似度（%W_TRANS%）...
cd /d "%TRANS_DIR%"
"%PY_TRANS%" transoss_inference.py --config_file configs\hoss_transoss.yml --weight "%W_TRANS%" --task_json "%TASK_JSON%" --out_prediction "%TRANS_SIM%\prediction_base.json" --save_sim "%TRANS_SIM%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] TransOSS 推理失败，错误码 %EC%
    pause
    exit /b %EC%
)

REM ---- 3. 融合 + 评测 ----
echo.
echo [12] 步骤 3/3：跨模型融合（method=%METHOD%, SDF权重=%W_A%）...
cd /d "%ROOT%"
"%PY%" scripts\fuse_sdfnet_transoss.py --sim_a "%SDF_SIM%\sim.pt" --meta_a "%SDF_SIM%\meta.json" --sim_b "%TRANS_SIM%\sim.pt" --meta_b "%TRANS_SIM%\meta.json" --task_json "%TASK_JSON%" --gt_json "%GT_JSON%" --out_dir "%OUT_DIR%" --method "%METHOD%" --weight_a %W_A%
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [12] 融合失败，错误码 %EC%
    pause
    exit /b %EC%
)

echo.
echo [12] 融合完成。结果文件：
echo     融合预测: %OUT_DIR%\prediction_weighted.json / prediction_rrf.json
echo     对比汇总: %OUT_DIR%\fusion_report.txt
echo     上方对比表含 SDF-Net 单模型 / TransOSS 单模型 / 加权融合 / RRF 融合的
echo     O2S / S2O / O2O / 综合得分；保留增益明显的融合方案用于正式提交。
pause
