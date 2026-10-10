"""用训练好的 CycleGAN 生成翻译图像。

支持两种模式：
1. 对整个目录批量翻译（--mode dir）
2. 对赛题训练集翻译并生成增强数据，输出 identity_map 供 SDF-Net 使用（--mode augment）

用法（批量翻译目录）：
    python generate.py --mode dir \
        --input_dir cyclegan_data/trainA \
        --output_dir generated/fake_B \
        --direction AtoB \
        --checkpoint checkpoints/ship_os/netG_A.pth

用法（增强 SDF-Net 训练集）：
    python generate.py --mode augment \
        --train_csv ../赛题6-初赛/训练数据/labels_train.csv \
        --checkpoint_dir checkpoints/ship_os \
        --out_dir ../SDF-Net/data/HOSS_augmented

输出 augment 模式：
    out_dir/
    ├── optical_to_sar/  # 由 optical 生成的 fake sar，文件名同原图
    └── sar_to_optical/  # 由 sar 生成的 fake optical，文件名同原图
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import ResnetGenerator


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dir", "augment"], default="dir")
    # dir 模式
    p.add_argument("--input_dir", type=str, help="输入图像目录")
    p.add_argument("--output_dir", type=str, help="输出图像目录")
    p.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB",
                   help="A=optical, B=sar; AtoB=optical->sar, BtoA=sar->optical")
    # augment 模式
    p.add_argument("--train_csv", type=str, help="labels_train.csv")
    p.add_argument("--checkpoint_dir", type=str, help="checkpoints/ship_os 目录")
    p.add_argument("--out_dir", type=str, help="增强数据输出目录")
    # 通用
    p.add_argument("--checkpoint", type=str, help="单个生成器权重路径（dir 模式用）")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--gpu_ids", type=str, default="0")
    return p.parse_args()


def build_generator(checkpoint_path: str, device):
    g = ResnetGenerator(3, 3, 64, 9, nn.InstanceNorm2d)
    state = torch.load(checkpoint_path, map_location=device)
    g.load_state_dict(state)
    g.to(device).eval()
    return g


transform = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def translate_image(g, src_path: Path, dst_path: Path, device):
    img = Image.open(src_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = g(t)
    out = out.squeeze(0).cpu().float().clamp(-1, 1)
    out = (out + 1) / 2  # [-1,1] -> [0,1]
    arr = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(dst_path)


def run_dir_mode(args, device):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    g = build_generator(args.checkpoint, device)
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    print(f"翻译 {len(files)} 张图: {input_dir} -> {output_dir}")
    for i, f in enumerate(files, 1):
        translate_image(g, f, output_dir / f.name, device)
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}]")
    print("完成")


def run_augment_mode(args, device):
    csv_path = Path(args.train_csv)
    data_root = csv_path.parent
    out = Path(args.out_dir)
    o2s_dir = out / "optical_to_sar"
    s2o_dir = out / "sar_to_optical"
    o2s_dir.mkdir(parents=True, exist_ok=True)
    s2o_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir)
    # G_A: B->A (sar->optical), G_B: A->B (optical->sar)
    g_a = build_generator(str(ckpt_dir / "netG_A.pth"), device)
    g_b = build_generator(str(ckpt_dir / "netG_B.pth"), device)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_o2s = n_s2o = 0
    for r in rows:
        iid = r["image_id"]
        mod = r["modality"]
        src = data_root / f"{mod}/{iid}.tif"
        if not src.exists():
            continue
        if mod == "optical":
            # optical -> fake sar (G_B = A->B)
            translate_image(g_b, src, o2s_dir / f"{iid}.tif", device)
            n_o2s += 1
        elif mod == "sar":
            # sar -> fake optical (G_A = B->A)
            translate_image(g_a, src, s2o_dir / f"{iid}.tif", device)
            n_s2o += 1

    print(f"增强数据生成完成: {out}")
    print(f"  optical_to_sar: {n_o2s}")
    print(f"  sar_to_optical: {n_s2o}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}, mode={args.mode}")
    if args.mode == "dir":
        run_dir_mode(args, device)
    else:
        run_augment_mode(args, device)


if __name__ == "__main__":
    # 延迟导入 nn，避免 models.py 顶部循环依赖
    from torch import nn
    main()
