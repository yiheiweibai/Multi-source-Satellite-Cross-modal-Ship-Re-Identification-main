"""计算训练集各模态的 mean/std，用于模态独立归一化。

用法：
    python scripts/compute_stats.py --csv ../../question6-data/traindata/labels_train.csv

输出 optical_mean/std 与 sar_mean/std，可填入 config/default.yaml。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="", help="labels.csv 路径；留空则用 --config 的 data_root/labels.csv")
    p.add_argument("--config", type=str, default="config/default.yaml", help="读取 data_root 的配置")
    p.add_argument("--max_samples", type=int, default=0, help="0=全部")
    return p.parse_args()


def _resolve_csv(args) -> str:
    if args.csv:
        return args.csv
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfgd = yaml.safe_load(f) or {}
    return str(Path(cfgd.get("data_root", "data")) / "labels.csv")


def main() -> None:
    args = parse_args()
    csv_path = _resolve_csv(args)
    print(f"labels.csv: {csv_path}")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    root = Path(csv_path).parent

    by_mod = {"optical": [], "sar": []}
    for r in rows:
        by_mod[r["modality"]].append(r["image_path"])
    if args.max_samples > 0:
        by_mod = {k: v[:args.max_samples] for k, v in by_mod.items()}

    print(f"optical: {len(by_mod['optical'])}, sar: {len(by_mod['sar'])}")

    stats = {}
    for mod, paths in by_mod.items():
        if not paths:
            continue
        sums = np.zeros(3, dtype=np.float64)
        sq_sums = np.zeros(3, dtype=np.float64)
        n = 0
        for p in paths:
            img = np.asarray(Image.open(root / p).convert("RGB"), dtype=np.float64) / 255.0
            sums += img.reshape(-1, 3).sum(axis=0)
            sq_sums += (img.reshape(-1, 3) ** 2).sum(axis=0)
            n += img.shape[0] * img.shape[1]
        mean = sums / n
        std = np.sqrt(sq_sums / n - mean ** 2)
        stats[mod] = (mean.tolist(), std.tolist())
        print(f"\n{mod}:")
        print(f"  mean: {[round(x, 4) for x in mean]}")
        print(f"  std:  {[round(x, 4) for x in std]}")

    print("\n# 填入 config/default.yaml:")
    for mod, (m, s) in stats.items():
        print(f"{mod}_mean: [{', '.join(f'{x:.4f}' for x in m)}]")
        print(f"{mod}_std: [{', '.join(f'{x:.4f}' for x in s)}]")


if __name__ == "__main__":
    main()
