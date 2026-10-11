#!/bin/bash
# AutoDL 正式测试集推理（CycleGAN 增强成员，bs32 ep30 best, mAP 74.5%）
# 前提：已通过网页上传 /root/autodl-tmp/test.zip（内含 task.json + query/ + gallery/）
set -e
TMP=/root/autodl-tmp
SDF=$TMP/SDF-Net

echo "========== [1/2] 解压测试数据 =========="
cd "$TMP"
if [ ! -d "$TMP/test_data/query" ]; then
    unzip -o -q test.zip -d test_data_raw
    # 兼容 zip 内是否包了一层目录
    if [ -d test_data_raw/query ]; then
        cp -r test_data_raw/* test_data/
    else
        INNER=$(find test_data_raw -maxdepth 2 -type d -name query -printf '%h\n' | head -1)
        cp -r "$INNER"/* test_data/
    fi
fi
echo "query: $(ls $TMP/test_data/query | wc -l), gallery: $(ls $TMP/test_data/gallery | wc -l)"

echo "========== [2/2] 推理（纯特征，无 rerank/QE/TTA）=========="
cd "$SDF"
python3 -u "$TMP/scripts/sdfnet_inference.py" \
    --config_file configs/SDF-Net-autodl.yml \
    --weight logs/SDF-Net-finetune/best.pth \
    --task_json "$TMP/test_data/task.json" \
    --out_prediction "$TMP/prediction_cyclegan.json" \
    --save_sim "$TMP/sims/cyclegan_ep30" \
    --batch_size 32 --num_workers 8

echo "========== 完成 =========="
ls -lh "$TMP/prediction_cyclegan.json" "$TMP/sims/cyclegan_ep30/"
