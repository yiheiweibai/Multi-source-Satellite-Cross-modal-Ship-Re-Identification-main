#!/bin/bash
# AutoDL 一键部署并启动 SDF-Net 微调（CycleGAN 增强数据）。
# 前提：
#   /root/autodl-tmp/optical, /root/autodl-tmp/sar         原始 7422 张图
#   /root/autodl-tmp/aug/fake_sar, fake_optical            CycleGAN ep30 生成图
#   /root/autodl-tmp/meta/{labels_train.csv,labels.csv,local_val_task.json}
#   /root/autodl-tmp/SDF-Net/                              SDF-Net 代码
#   /root/autodl-tmp/scripts/{sdfnet_prepare_data.py,integrate_cyclegan.py}
set -e
TMP=/root/autodl-tmp
SDF=$TMP/SDF-Net

echo "========== [1/5] Python 依赖 =========="
pip3 install -q timm==1.0.25 yacs==0.1.8 opencv-python-headless thop scipy tifffile 2>&1 | tail -2
python3 -c "import timm, yacs, cv2, scipy, tifffile; print('deps OK')"

echo "========== [2/5] 预训练权重 =========="
WDIR=$SDF/logs/SDF-Net
mkdir -p "$WDIR"
W=$WDIR/SDF-Net_256.pth
if [ ! -f "$W" ]; then
    echo "从 HuggingFace 下载 ..."
    curl -fsSL -o "$W" "https://huggingface.co/Chenfree233/SDF-Net/resolve/main/SDF-Net.pth" \
        || { echo "HF 失败，尝试 hf-mirror ..."; curl -fsSL -o "$W" "https://hf-mirror.com/Chenfree233/SDF-Net/resolve/main/SDF-Net.pth"; }
fi
ls -lh "$W"

# 官方权重在 256x128 下训练（pos_embed=130 tokens），本配置 256x256 需要 258，需插值
ORIG=$WDIR/SDF-Net_official_256x128.pth
if python3 -c "import torch,sys; sys.exit(0 if tuple(torch.load('$W',map_location='cpu').get('base.pos_embed',torch.empty(1,0)).shape)==(1,258,768) else 1)" 2>/dev/null; then
    echo "pos_embed 已是 258 tokens，跳过转换"
else
    echo "转换 pos_embed 130 -> 258 ..."
    [ -f "$ORIG" ] || cp "$W" "$ORIG"
    python3 "$TMP/scripts/sdfnet_resize_posembed.py" \
        --in_weight "$ORIG" --out_weight "$W"
fi

echo "========== [3/5] 准备 HOSS 基础数据 =========="
# prepare 脚本按 train_csv 同级找 optical/sar，建软链接
ln -sfn "$TMP/optical" "$TMP/meta/optical"
ln -sfn "$TMP/sar" "$TMP/meta/sar"
mkdir -p "$SDF/data"
python3 "$TMP/scripts/sdfnet_prepare_data.py" \
    --labels_train "$TMP/meta/labels_train.csv" \
    --labels_full  "$TMP/meta/labels.csv" \
    --task         "$TMP/meta/local_val_task.json" \
    --out_dir      "$SDF/data/HOSS_base/HOSS"

echo "========== [4/5] 并入 CycleGAN fake 图 =========="
python3 "$TMP/scripts/integrate_cyclegan.py" \
    --hoss_dir "$SDF/data/HOSS_base/HOSS" \
    --aug_dir  "$TMP/aug/HOSS_augmented" \
    --out_dir  "$SDF/data/HOSS"

echo "========== [5/5] 启动微调（后台）=========="
cd "$SDF"
nohup python3 -u train.py \
    --config_file configs/SDF-Net-autodl.yml \
    > "$TMP/sdfnet_train.log" 2>&1 &
echo "训练 PID: $!"
sleep 10
echo "========== 启动日志 =========="
tail -20 "$TMP/sdfnet_train.log" || true
