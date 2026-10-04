@echo off
chcp 65001 >nul
REM 11_transoss_ab.bat - 一键消融验证：base / preprocess / rerank+QE / mixed 四组合横向对比
REM 前提：
REM   1. 已运行 08_transoss_setup.bat 完成环境准备（Hoss-ReID 存在）
REM   2. 已运行 07_transoss_pipeline.bat 完成训练（权重 logs/competition_transoss/transformer_200.pth）
REM   3. 赛题数据已同步（../赛题6-初赛/训练数据/labels.csv）
REM 验证集：优先复用已存在的 local_val_task.json（不重新划分，保证与 09/10 结果横向可比）；
REM         不存在时才调用 build_local_val.py 生成一次（会重写 labels_train.csv，请确认已备份）
cd /d "%~dp0"

set ROOT=%~dp0
set HOSS_DIR=%ROOT%Hoss-ReID
set PY=%ROOT%.venv\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json

REM ---- 0. 环境检查 ----
if not exist "%PY%" (
    echo [11] 未找到 .venv，请先运行 01_setup.bat 创建虚拟环境。
    pause
    exit /b 1
)
if not exist "%HOSS_DIR%\transoss_inference.py" (
    echo [11] 未找到 Hoss-ReID\transoss_inference.py，请先运行 08_transoss_setup.bat。
    pause
    exit /b 1
)
if not exist "%HOSS_DIR%\logs\competition_transoss\transformer_200.pth" (
    echo [11] 未找到训练权重 transformer_200.pth，请先运行 07_transoss_pipeline.bat 完成训练。
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\labels.csv" (
    echo [11] 未找到全量标签 labels.csv，请确认赛题数据已同步。
    pause
    exit /b 1
)

REM ---- 1. 验证集：存在则复用，不存在才生成一次 ----
if exist "%TASK_JSON%" (
    echo [11] 复用已有验证集 local_val_task.json（不重新划分，保证横向可比）
) else (
    echo [11] 未找到 local_val_task.json，将调用 build_local_val.py 生成一次。
    echo     注意：build_local_val.py 会重写 labels_train.csv 为 80%% 身份子集，
    echo     请确认已备份（09 脚本首次运行会生成 labels_train.backup.csv）。
    "%PY%" scripts\build_local_val.py --csv "%DATA_DIR%\labels.csv" --out_dir "%DATA_DIR%" --val_ratio 0.2 --seed 42
    if errorlevel 1 (
        echo [11] 验证集生成失败，请查看上方错误。
        pause
        exit /b 1
    )
)

REM ---- 2. 一键消融：4 组合推理 + 评测 + 对比汇总表 ----
echo.
echo [11] 一键消融验证（base / preprocess / rerank+QE / mixed）...
"%PY%" scripts\transoss_ablation.py --task_json "%TASK_JSON%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [11] 消融验证失败，错误码 %EC%
    pause
    exit /b %EC%
)
echo [11] 消融完成：pred_ab_base.json / pred_ab_preprocess.json / pred_ab_rerankqe.json / pred_ab_mixed.json
echo     上方汇总表为同一验证集横向对比，只保留相对 baseline 为正增益的组合项。
pause
