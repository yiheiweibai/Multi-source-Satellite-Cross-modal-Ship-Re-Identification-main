#!/bin/bash
# base75 权重在正式测试集推理：导出 sim 矩阵 + top100 JSON（供 RRF 融合）
set -e
TMP=/root/autodl-tmp
SDF=$TMP/SDF-Net

echo "===== 数据核对 ====="
echo "query: $(ls $TMP/test_data/query | wc -l), gallery: $(ls $TMP/test_data/gallery | wc -l)"
ls -lh "$SDF/logs/SDF-Net-base75/best.pth"

echo "===== base75 测试集推理 (top100 + save_sim) ====="
cd "$SDF"
python3 -u "$TMP/scripts/sdfnet_inference.py" \
    --config_file configs/SDF-Net-autodl-base75.yml \
    --weight logs/SDF-Net-base75/best.pth \
    --task_json "$TMP/test_data/task.json" \
    --out_prediction "$TMP/top100_base75.json" \
    --save_sim "$TMP/sims/test_base75" \
    --batch_size 32 --num_workers 8 --topk 100

echo "===== 完成 ====="
ls -lh "$TMP/top100_base75.json" "$TMP/sims/test_base75/"
