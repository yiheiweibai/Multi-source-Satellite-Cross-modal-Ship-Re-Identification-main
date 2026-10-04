@echo off
chcp 65001 >nul
rem ============================================================
rem [优化节奏] 本脚本属于"零训练成本优先"阶段：
rem   1) 先看下方输出的多 checkpoint / RRF 融合对比汇总表；
rem   2) 再配合 --preprocess（SAR 预处理）、TTA、rerank、QE 等
rem      推理端增强逐项试，只保留正增益项；
rem   3) 对比表达标后再对测试集推理提交；
rem   4) 不达标才考虑重训兜底（伪标签自训练 1 轮约 3.9 小时、
rem      分辨率微调等），可基于已有 transformer_*.pth 续训。
rem ============================================================
cd /d "%~dp0"

echo [10] 多 checkpoint 选优 + RRF 融合

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv，请先运行 08_transoss_setup.bat 初始化环境
    pause
    exit /b 1
)

if not exist "Hoss-ReID\transoss_inference.py" (
    echo [错误] 未找到 Hoss-ReID\transoss_inference.py，请先运行 08_transoss_setup.bat 克隆 TransOSS
    pause
    exit /b 1
)

if not exist "Hoss-ReID\logs\competition_transoss" (
    echo [错误] 未找到权重目录 Hoss-ReID\logs\competition_transoss，请先运行 07_transoss_pipeline.bat 完成训练
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\transoss_checkpoint_fusion.py --task_json "..\赛题6-初赛\训练数据\local_val_task.json"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [错误] 融合脚本执行失败，错误码 %EC%
    pause
    exit /b %EC%
)
echo [完成] 融合结果已生成：pred_ckpt_*.json 与 pred_fusion.json
pause
