@echo off
chcp 65001 >nul
REM 09_local_val_transoss.bat - 一键本地验证 TransOSS：划分验证集 -> TransOSS 推理 -> 赛题指标评测
REM 用法：在 16GB 目标机（已跑通 07 的机器）双击本脚本，全程自动，结果自动对比自研基线。
REM 前提：
REM   1. 已同步仓库代码与数据（ship_reid_system 同级存在 赛题6-初赛/训练数据/labels.csv 全量标签）
REM   2. 已运行 08_transoss_setup.bat 完成环境准备
REM   3. 已运行 07_transoss_pipeline.bat 完成训练（权重 logs/competition_transoss/transformer_200.pth）
REM 注意：本脚本会重新划分验证集并重写 labels_train.csv（首次运行前自动备份到 labels_train.backup.csv）
cd /d "%~dp0"

set ROOT=%~dp0
set HOSS_DIR=%ROOT%Hoss-ReID
set PY=%ROOT%.venv\Scripts\python.exe
set DATA_DIR=%ROOT%..\赛题6-初赛\训练数据

REM ---- 0. 环境检查 ----
if not exist "%PY%" (
    echo [09] 未找到 .venv，请先运行 01_setup.bat 创建虚拟环境。
    pause
    exit /b 1
)
if not exist "%HOSS_DIR%\transoss_inference.py" (
    echo [09] 未找到 Hoss-ReID\transoss_inference.py，请先运行 08_transoss_setup.bat。
    pause
    exit /b 1
)
if not exist "%HOSS_DIR%\logs\competition_transoss\transformer_200.pth" (
    echo [09] 未找到训练权重 transformer_200.pth，请先运行 07_transoss_pipeline.bat 完成训练。
    pause
    exit /b 1
)
if not exist "%DATA_DIR%\labels.csv" (
    echo [09] 未找到全量标签 labels.csv，请确认赛题数据已同步。
    pause
    exit /b 1
)

REM ---- 1. 备份 labels_train.csv（build_local_val 会重写为 80%% 身份子集）----
if exist "%DATA_DIR%\labels_train.csv" (
    if not exist "%DATA_DIR%\labels_train.backup.csv" (
        copy /Y "%DATA_DIR%\labels_train.csv" "%DATA_DIR%\labels_train.backup.csv" >nul
        echo [09] 已备份 labels_train.csv -^> labels_train.backup.csv
    ) else (
        echo [09] 已有备份 labels_train.backup.csv，跳过备份
    )
)

REM ---- 2. 划分验证集（20%% 身份，seed 42）----
echo.
echo [1/3] 从训练集划分验证集...
"%PY%" scripts/build_local_val.py --csv "../赛题6-初赛/训练数据/labels.csv" --out_dir "../赛题6-初赛/训练数据" --val_ratio 0.2 --seed 42
if errorlevel 1 (
    echo [09] 验证集划分失败，请查看上方错误。
    pause
    exit /b 1
)

REM ---- 3. TransOSS 推理（在 Hoss-ReID 仓库内执行，默认无后处理）----
echo.
echo [2/3] TransOSS 在本地验证集上推理...
cd /d "%HOSS_DIR%"
"..\.venv\Scripts\python.exe" transoss_inference.py ^
    --config_file configs/hoss_transoss_competition.yml ^
    --weight logs/competition_transoss/transformer_200.pth ^
    --task_json "../../赛题6-初赛/训练数据/local_val_task.json" ^
    --out_prediction ../pred_local_val.json
if errorlevel 1 (
    echo [09] TransOSS 推理失败，请查看上方错误。
    pause
    exit /b 1
)

REM ---- 4. 按赛题指标评测 ----
echo.
echo [3/3] 评测本地验证集...
cd /d "%ROOT%"
"%PY%" evaluate.py --submission ^
    --prediction pred_local_val.json ^
    --task "../赛题6-初赛/训练数据/local_val_task.json" ^
    --gt "../赛题6-初赛/训练数据/local_val_gt.json"
if errorlevel 1 (
    echo [09] 评测失败，请查看上方错误。
    pause
    exit /b 1
)

echo.
echo [09] 本地验证完成！pred_local_val.json 已生成（ship_reid_system 目录）。
echo     上方综合得分可与自研基线对比：
echo       自研无后处理基线：综合 0.1162（O2S 0.1055 / S2O 0.1094 / O2O 0.1954）
echo       自研加后处理：    综合 0.0948（后处理当前无正增益，默认关闭）
echo     本次默认直接检索（无后处理），如需测后处理，修改第 3 步追加：--tta --rerank --qe --cluster
pause
