@echo off
chcp 65001 >nul
REM 08_transoss_setup.bat - 一键准备 TransOSS 开源模型运行环境（可在另一台 4070 Ti SUPER 机器上运行）
REM 前提：已同步本仓库代码与赛题数据（赛题6-初赛/训练数据/labels_train.csv），且已运行 01_setup.bat
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [08] 未找到 .venv，请先运行 01_setup.bat 创建虚拟环境并安装基础依赖。
    pause
    exit /b 1
)

echo [08] 开始准备 TransOSS 运行环境（克隆仓库 / 装依赖 / 数据转换 / 占位目录 / 配置生成）...
.venv\Scripts\python.exe scripts/transoss_setup.py
if errorlevel 1 (
    echo.
    echo [08] 准备失败，请按上方错误提示处理。
    pause
    exit /b 1
)

echo.
echo [08] 准备完成，可运行 07_transoss_pipeline.bat 开始训练与推理。
pause
