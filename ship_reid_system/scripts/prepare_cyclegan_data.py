"""从赛题训练数据准备 CycleGAN 训练数据（trainA=optical, trainB=sar）。

用法：
    python scripts/prepare_cyclegan_data.py \
        --train_csv ../赛题6-初赛/训练数据/labels_train.csv \
        --out_dir cyclegan_data

输出：
    cyclegan_data/
    ├── trainA/  # optical 图像（复制原始 .tif）
    └── trainB/  # sar 图像（复制原始 .tif）
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="labels_train.csv")
    p.add_argument("--out_dir", type=str, required=True, help="输出 cyclegan_data 目录")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.train_csv)
    data_root = csv_path.parent
    out = Path(args.out_dir)
    train_a = out / "trainA"
    train_b = out / "trainB"
    train_a.mkdir(parents=True, exist_ok=True)
    train_b.mkdir(parents=True, exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_a = n_b = 0
    for r in rows:
        iid = r["image_id"]
        mod = r["modality"]
        src = data_root / f"{mod}/{iid}.tif"
        if not src.exists():
            continue
        if mod == "optical":
            shutil.copy2(src, train_a / f"{iid}.tif")
            n_a += 1
        elif mod == "sar":
            shutil.copy2(src, train_b / f"{iid}.tif")
            n_b += 1

    print(f"准备完成: {out}")
    print(f"  trainA (optical): {n_a}")
    print(f"  trainB (sar): {n_b}")


if __name__ == "__main__":
    main()
