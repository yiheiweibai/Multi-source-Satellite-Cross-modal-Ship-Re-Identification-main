@echo off
chcp 65001 >nul
REM 07_transoss_pipeline.bat - TransOSS 全流程（数据转换 -> 微调 -> 推理+后处理 -> 提交）
REM 前提：已运行 08_transoss_setup.bat（克隆 Hoss-ReID、装依赖、数据转换、占位目录、配置生成）
REM 训练内置评估使用占位数据，评估数字不可信；最终精度以竞赛测试集推理 + 本地验证协议为准。
cd /d "%~dp0"

set HOSS_DIR=%~dp0Hoss-ReID
set PY=%~dp0.venv\Scripts\python.exe

if not exist "%HOSS_DIR%\configs\hoss_transoss_competition.yml" (
    echo [07] 未找到 TransOSS 配置，请先运行 08_transoss_setup.bat 完成环境准备。
    pause
    exit /b 1
)

REM ---- 1. 数据转换（幂等，已存在则跳过） ----
echo [1/4] 转换训练数据为 TransOSS 格式...
"%PY%" scripts/transoss_prepare_data.py ^
    --csv "../赛题6-初赛/训练数据/labels_train.csv" ^
    --out_dir "../transoss_data/HOSS/bounding_box_train"
if errorlevel 1 ( pause & exit /b 1 )

REM ---- 2. 微调 TransOSS（自动执行，200 epochs，约 4 小时） ----
echo.
echo [2/4] 微调 TransOSS（在 Hoss-ReID 仓库内，200 epochs）...
cd /d "%HOSS_DIR%"
"..\.venv\Scripts\python.exe" train.py --config_file configs/hoss_transoss_competition.yml
if errorlevel 1 (
    echo [07] 训练失败。常见原因：权重缺失（08 检查过）/ OOM / 依赖不全。
    pause
    exit /b 1
)

REM ---- 3. 推理 + 后处理（默认无后处理；修复后如需重测后处理可加 --tta --rerank --qe） ----
echo.
echo [3/4] TransOSS 推理（生成 prediction_transoss.json）...
"..\.venv\Scripts\python.exe" transoss_inference.py ^
    --config_file configs/hoss_transoss_competition.yml ^
    --weight logs/competition_transoss/transformer_200.pth ^
    --task_json "../赛题6-初赛/初赛测试数据/task.json" ^
    --out_prediction ../prediction_transoss.json
if errorlevel 1 (
    echo [07] 推理失败。
    pause
    exit /b 1
)

echo.
echo [4/4] 完成！prediction_transoss.json 已生成（位于项目根目录），可提交。
echo     若修复后的后处理（TTA/rerank/QE）在本地验证有正增益，可追加参数重跑：
echo       --tta --rerank --qe
pause
