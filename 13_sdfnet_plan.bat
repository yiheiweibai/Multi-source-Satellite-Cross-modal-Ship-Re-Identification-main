@echo off
REM 13_sdfnet_plan.bat - SDF-Net experiment plan entry (stage-by-stage comparison)
REM   Combinations (all reuse the fixed local_val_task.json; fine-tune runs only once):
REM     A1 : SDF-Net official weights, base
REM     A2 : SDF-Net official weights + rerankqe (per query_type post-process)
REM     B1 : SDF-Net after fine-tuning, base
REM     B2 : SDF-Net after fine-tuning + rerankqe
REM     C  : TransOSS existing weights + fixed rerankqe
REM     D1~D3 : SDF-Net(fine-tuned) x TransOSS score fusion, SDF weight 0.5/0.6/0.7
REM     D4 : same pair with RRF fusion
REM   A comparison document is written at the end:
REM     experiments\exp_005_sdfnet_plan_compare.md
REM Configurable environment variables (override the defaults):
REM   SDF_DATA_DIR      competition data dir (default: h:\Ship-Re-Identification\question6-data\traindata)
REM   SDF_EPOCHS        fine-tune epochs, default 60
REM   SDF_IMS_PER_BATCH fine-tune batch size, default 32 (16GB GPU: use 16 on OOM)
REM   SDF_FINETUNE_DIR  fine-tune output dir, default logs\SDF-Net-finetune
REM   SDF_WEIGHT        official weight path, default SDF-Net\logs\SDF-Net\SDF-Net_256.pth
REM   TRANS_WEIGHT      TransOSS weight path, default Hoss-ReID\logs\competition_transoss\transformer_200.pth
REM Prerequisites:
REM   1. SDF-Net side prepared by 12_sdfnet_setup.bat (.venv-sdfnet / repo / weights / HOSS data)
REM   2. TransOSS side prepared by 08_transoss_setup.bat; if missing, C/D are skipped automatically
REM   3. local_val_task.json / local_val_gt.json exist
cd /d "%~dp0"

set ROOT=%~dp0
set PY=%ROOT%.venv-sdfnet\Scripts\python.exe
if defined SDF_DATA_DIR (set DATA_DIR=%SDF_DATA_DIR%) else (set DATA_DIR=h:\Ship-Re-Identification\question6-data\traindata)
set TASK_JSON=%DATA_DIR%\local_val_task.json
set GT_JSON=%DATA_DIR%\local_val_gt.json
set SDF_DIR=%ROOT%SDF-Net
set WEIGHT=%SDF_DIR%\logs\SDF-Net\SDF-Net_256.pth

if defined SDF_EPOCHS (set EPOCHS=%SDF_EPOCHS%) else (set EPOCHS=60)
if defined SDF_IMS_PER_BATCH (set IMS=%SDF_IMS_PER_BATCH%) else (set IMS=32)
if defined SDF_FINETUNE_DIR (set FT_DIR=%SDF_FINETUNE_DIR%) else (set FT_DIR=%ROOT%logs\SDF-Net-finetune)
if defined SDF_WEIGHT (set W_SDF=%SDF_WEIGHT%) else (set W_SDF=%WEIGHT%)
if defined TRANS_WEIGHT (set W_TRANS=%TRANS_WEIGHT%) else (set W_TRANS=%ROOT%Hoss-ReID\logs\competition_transoss\transformer_200.pth)

REM ---- 0. environment checks ----
if not exist "%PY%" (
    echo [13] .venv-sdfnet not found, please run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)
if not exist "%W_SDF%" (
    echo [13] SDF-Net official weights not found: %W_SDF%
    echo      please run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)
if not exist "%TASK_JSON%" (
    echo [13] local_val_task.json not found: %TASK_JSON%
    pause
    exit /b 1
)
if not exist "%GT_JSON%" (
    echo [13] local_val_gt.json not found: %GT_JSON%
    pause
    exit /b 1
)
if not exist "%SDF_DIR%\configs\SDF-Net.yml" (
    echo [13] SDF-Net config not found, please run 12_sdfnet_setup.bat first.
    pause
    exit /b 1
)

echo.
echo [13] plan start:
echo      fine-tune epochs=%EPOCHS%   batch=%IMS%   out=%FT_DIR%
echo      official weights=%W_SDF%
echo      TransOSS weights=%W_TRANS%
echo      (combinations needing a GPU or missing weights are skipped and reported in the log)
echo.

REM ---- 1. run the orchestrator (stages + evaluation + comparison document) ----
"%PY%" scripts\sdfnet_plan.py --epochs %EPOCHS% --ims_per_batch %IMS% --ft_dir "%FT_DIR%" --sdf_official_weight "%W_SDF%" --trans_weight "%W_TRANS%" --task_json "%TASK_JSON%" --gt_json "%GT_JSON%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
    echo [13] plan failed with exit code %EC% (see sims\plan\run.log)
    pause
    exit /b %EC%
)

echo.
echo [13] plan finished. Outputs:
echo     comparison doc: %ROOT%experiments\exp_005_sdfnet_plan_compare.md
echo     intermediates:  %ROOT%sims\plan\ (pred_*.json / sim.pt / run.log)
pause