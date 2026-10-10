#!/bin/bash
# AutoDL 上训练 CycleGAN 的脚本。
# 使用方法：在 AutoDL 实例的 Jupyter/终端中运行：
#   bash autodl_train.sh
#
# 前置条件：
#   1. 已将 cyclegan/ 代码上传到 /root/autodl-tmp/cyclegan
#   2. 已将训练数据上传到 /root/autodl-tmp/cyclegan_data（trainA=optical, trainB=sar）
#      用本地 scripts/prepare_cyclegan_data.py 生成后上传
set -e

cd /root/autodl-tmp/cyclegan

# 依赖（AutoDL 的 PyTorch 镜像通常已含 torch/torchvision，只需补 tifffile）
pip install tifffile pillow -q

echo "=========================================="
echo "CycleGAN 训练开始"
echo "数据: /root/autodl-tmp/cyclegan_data"
echo "=========================================="

python train.py \
    --dataroot /root/autodl-tmp/cyclegan_data \
    --name ship_os \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --batch_size 4 \
    --lr 0.0002 \
    --lambda_A 10.0 \
    --lambda_B 10.0 \
    --lambda_identity 0.5 \
    --save_epoch_freq 20 \
    --print_freq 200

echo "=========================================="
echo "训练完成。权重保存在 checkpoints/ship_os/"
echo "下载 netG_A.pth 和 netG_B.pth 到本地"
echo "=========================================="
ls -lh checkpoints/ship_os/
