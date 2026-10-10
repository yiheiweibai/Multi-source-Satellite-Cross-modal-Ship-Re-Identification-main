@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM ============================================================
REM CycleGAN ???????????? + AutoDL ?? + ?????
REM ============================================================
REM ???
REM   1. ???? CycleGAN ?????trainA=optical, trainB=sar?
REM   2. [??] ?? cyclegan_data/ ? cyclegan/ ? AutoDL??? autodl_train.sh
REM   3. [??] ?? checkpoints/ship_os/ ??? cyclegan/checkpoints/ship_os/
REM   4. ?????????O->S, S->O?
REM   5. ????? SDF-Net ????
REM ============================================================

set ROOT=%~dp0
cd /d "%ROOT%"

REM ---- 1. ?? CycleGAN ???? ----
echo [1/5] ?? CycleGAN ????...
python scripts/prepare_cyclegan_data.py ^
    --train_csv "../??6-??/????/labels_train.csv" ^
    --out_dir cyclegan_data
if errorlevel 1 ( echo prepare_cyclegan_data ?? & pause & exit /b 1 )

echo.
echo ============================================================
echo [2/5] ?????? AutoDL ????
echo   - ?? cyclegan_data/  -^> /root/autodl-tmp/cyclegan_data
echo   - ?? cyclegan/       -^> /root/autodl-tmp/cyclegan
echo   - ? AutoDL ??: cd /root/autodl-tmp/cyclegan ^&^& bash autodl_train.sh
echo   - ?? checkpoints/ship_os/ ??? cyclegan/checkpoints/ship_os/
echo ============================================================
echo.
pause

REM ---- 4. ?????? ----
echo [4/5] ???????O-^>S, S-^>O?...
python cyclegan/generate.py --mode augment ^
    --train_csv "../??6-??/????/labels_train.csv" ^
    --checkpoint_dir cyclegan/checkpoints/ship_os ^
    --out_dir ../SDF-Net/data/HOSS_augmented
if errorlevel 1 ( echo generate ?? & pause & exit /b 1 )

REM ---- 5. ??? SDF-Net ???? ----
echo [5/5] ??????? SDF-Net ???...
if not exist "../SDF-Net/data/HOSS/identity_map.json" (
    echo ???? 12_sdfnet_setup.bat ???? HOSS ??
    pause & exit /b 1
)
python scripts/integrate_cyclegan.py ^
    --hoss_dir ../SDF-Net/data/HOSS ^
    --aug_dir ../SDF-Net/data/HOSS_augmented ^
    --out_dir ../SDF-Net/data/HOSS_aug
if errorlevel 1 ( echo integrate_cyclegan ?? & pause & exit /b 1 )

echo.
echo ============================================================
echo ??????????????: ../SDF-Net/data/HOSS_aug/
echo ?? SDF-Net ?? SDF-Net.yml ? DATASETS.ROOT_DIR ?? HOSS_aug
echo ============================================================
pause
