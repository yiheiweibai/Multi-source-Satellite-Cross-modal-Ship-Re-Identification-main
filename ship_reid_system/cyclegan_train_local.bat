@echo off
chcp 65001 >nul
REM ?? GPU ?? CycleGAN?4070 Ti SUPER?
REM ?????? prepare_cyclegan_data.py???????
set ROOT=%~dp0
cd /d "%ROOT%cyclegan"

python train.py ^
    --dataroot ../cyclegan_data ^
    --name ship_os ^
    --n_epochs 100 ^
    --n_epochs_decay 100 ^
    --batch_size 4 ^
    --lr 0.0002 ^
    --lambda_A 10.0 ^
    --lambda_B 10.0 ^
    --lambda_identity 0.5 ^
    --save_epoch_freq 20 ^
    --print_freq 200
pause
