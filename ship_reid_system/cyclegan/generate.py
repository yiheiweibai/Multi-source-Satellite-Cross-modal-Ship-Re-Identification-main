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

用法（增强 SDF-Net 训练集，AutoDL 任意工作目录下均可执行）：
    # 推荐：--dataroot 兼作路径基准，其余相对路径均相对于它解析；也可用绝对路径
    python generate.py --mode augment \
        --dataroot /root/autodl-tmp/cyclegan_data \
        --train_csv 赛题6-初赛/训练数据/labels_train.csv \
        --checkpoint_dir checkpoints/ship_os \
        --out_dir ../SDF-Net/data/HOSS_augmented
    # 本地旧用法兼容（不传 --dataroot，相对路径按当前工作目录解析）
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
    # 路径基准（保证 AutoDL 任意工作目录下相对路径不失效）
    p.add_argument("--dataroot", type=str, default=None,
                   help="数据根目录：1) augment 模式作为 optical/sar 图像目录的根；2) 未指定 --base_dir 时兼作所有相对路径的解析基准")
    p.add_argument("--base_dir", type=str, default=None,
                   help="相对路径解析基准目录（--train_csv/--out_dir/--checkpoint_dir/--input_dir/--output_dir/--checkpoint 的相对写法均相对于它解析）。默认取 --dataroot；两者均未提供时取当前工作目录（保持本地旧用法兼容）")
    # 通用
    p.add_argument("--checkpoint", type=str, help="单个生成器权重路径（dir 模式用）")
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--gpu_ids", type=str, default="0")
    return p.parse_args()


def get_base_dir(args) -> Path:
    """返回相对路径解析基准：--base_dir > --dataroot > 当前工作目录。"""
    if args.base_dir:
        return Path(args.base_dir).resolve()
    if args.dataroot:
        return Path(args.dataroot).resolve()
    return Path.cwd()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """将用户输入路径规范化为绝对路径：
    - 已是绝对路径（含盘符或以 / 开头）→ 原样使用；
    - 相对路径 → 相对 base_dir 解析（不再依赖进程当前工作目录）。
    """
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


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
    base = get_base_dir(args)
    input_dir = resolve_path(args.input_dir, base)
    output_dir = resolve_path(args.output_dir, base)
    output_dir.mkdir(parents=True, exist_ok=True)
    g = build_generator(str(resolve_path(args.checkpoint, base)), device)
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    print(f"翻译 {len(files)} 张图: {input_dir} -> {output_dir}")
    for i, f in enumerate(files, 1):
        translate_image(g, f, output_dir / f.name, device)
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}]")
    print("完成")


def run_augment_mode(args, device):
    base = get_base_dir(args)
    csv_path = resolve_path(args.train_csv, base)
    # 图像数据根目录推导（含 optical/ sar/ 子目录）：
    #   1) 显式 --dataroot → 直接使用；
    #   2) 未提供时回退 csv 所在目录（保持本地旧用法：csv 与 optical/sar 同级的目录）
    data_root = Path(args.dataroot).resolve() if args.dataroot else csv_path.parent
    if not (data_root / "optical").is_dir() and not (data_root / "sar").is_dir() \
            and csv_path.parent != data_root:
        fallback = csv_path.parent
        print(f"[generate] 警告: --dataroot={data_root} 下未找到 optical/sar 目录，"
              f"自动回退使用 csv 所在目录 {fallback} 作为数据根")
        data_root = fallback
    out = resolve_path(args.out_dir, base)
    o2s_dir = out / "optical_to_sar"
    s2o_dir = out / "sar_to_optical"
    o2s_dir.mkdir(parents=True, exist_ok=True)
    s2o_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = resolve_path(args.checkpoint_dir, base)
    # train.py 约定：G_A: A->B (optical->sar)，G_B: B->A (sar->optical)
    g_o2s = build_generator(str(ckpt_dir / "netG_A.pth"), device)  # optical -> sar
    g_s2o = build_generator(str(ckpt_dir / "netG_B.pth"), device)  # sar -> optical

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
            # optical -> fake sar (G_A = A->B)
            translate_image(g_o2s, src, o2s_dir / f"{iid}.tif", device)
            n_o2s += 1
        elif mod == "sar":
            # sar -> fake optical (G_B = B->A)
            translate_image(g_s2o, src, s2o_dir / f"{iid}.tif", device)
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
