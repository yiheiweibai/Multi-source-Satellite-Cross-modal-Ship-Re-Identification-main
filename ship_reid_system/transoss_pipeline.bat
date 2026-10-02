@echo off
chcp 65001 >nul
REM transoss_pipeline.bat - TransOSS 全流程（数据转换 -> 微调 -> 推理+后处理 -> 提交）
REM 前提：已克隆 https://github.com/Alioth2000/Hoss-ReID 到本脚本同级目录
cd /d "%~dp0"

REM ---- 1. 数据转换 ----
echo [1/4] 转换训练数据为 TransOSS 格式...
.venv\Scripts\python.exe scripts/transoss_prepare_data.py ^
    --csv "../赛题6-初赛/训练数据/labels_train.csv" ^
    --out_dir "../transoss_data/HOSS/bounding_box_train"
if errorlevel 1 ( pause & exit /b 1 )

REM ---- 2. 微调 TransOSS ----
echo.
echo [2/4] 微调 TransOSS（在 Hoss-ReID 仓库内）...
REM 需将 transoss_config/hoss_transoss_competition.yml 复制到 Hoss-ReID/configs/
REM 然后在 Hoss-ReID 目录执行：
REM   python train.py --config_file configs/hoss_transoss_competition.yml
echo  请手动进入 Hoss-ReID 目录执行训练，完成后按回车继续
pause

REM ---- 3. 推理 + 后处理 ----
echo.
echo [3/4] TransOSS 推理 + 后处理...
REM transoss_inference.py 需复制到 Hoss-ReID 根目录运行
echo  请手动在 Hoss-ReID 目录执行推理，完成后按回车继续
echo  命令示例：
echo    python transoss_inference.py --config_file configs/hoss_transoss_competition.yml ^
echo        --weight logs/competition_transoss/transformer_200.pth ^
echo        --task_json "../赛题6-初赛/初赛测试数据/task.json" ^
echo        --out_prediction ../prediction_transoss.json ^
echo        --tta --rerank --qe --cluster
pause

echo.
echo [4/4] 完成！prediction_transoss.json 可提交
pause
