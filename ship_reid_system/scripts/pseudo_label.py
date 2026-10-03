"""伪标签自训练：用训练好的模型对测试集 gallery 打伪标签，高置信样本加入训练集。

用法：
    python scripts/pseudo_label.py \
        --ckpt outputs/checkpoints/best.pth \
        --train_csv ../赛题6-初赛/训练数据/labels_train.csv \
        --task_json ../赛题6-初赛/初赛测试数据/task.json \
        --out_csv ../赛题6-初赛/训练数据/labels_pseudo.csv \
        --threshold 0.85

产出：labels_pseudo.csv（原始训练 + 高置信伪标签 gallery），可直接用于 train.py。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import ShipReIDCSVDataset, build_transforms
from data.dataset import MODALITY_NAME
from data.test_dataset import GalleryDataset, load_task
from inference import RetrievalEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-label gallery for self-training")
    p.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint")
    p.add_argument("--train_csv", type=str, required=True, help="训练集 labels.csv")
    p.add_argument("--task_json", type=str, required=True, help="测试集 task.json")
    p.add_argument("--out_csv", type=str, default="outputs/pseudo/labels_pseudo.csv", help="输出伪标签 CSV 路径（默认 outputs/pseudo/，避免写回官方数据目录造成路径混乱）")
    p.add_argument("--threshold", type=float, default=0.85, help="相似度阈值（仅保留高于此值的伪标签）")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)

    # 1. 加载训练集特征
    print("[1/4] 加载训练集并提取特征...")
    train_ds = ShipReIDCSVDataset(
        args.train_csv,
        transform=build_transforms(cfg, is_train=False),
        is_train=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    engine = RetrievalEngine(cfg, args.ckpt, device=args.device)
    train_feats = engine.extract(train_loader)  # (N, D)
    train_labels = train_ds.labels()            # (N,) 整数身份
    train_ship_ids = [s["ship_id"] for s in train_ds.samples]
    # 整数身份 -> ship_id 字符串
    id2ship = {}
    for lbl, sid in zip(train_labels, train_ship_ids):
        id2ship[int(lbl)] = sid
    print(f"  训练集: {len(train_ds)} 样本, {len(id2ship)} 身份")

    # 2. 加载测试集 gallery
    print("[2/4] 加载测试集 gallery 并提取特征...")
    task = load_task(args.task_json)
    gallery = task["gallery"]
    gallery_ds = GalleryDataset(gallery, transform=build_transforms(cfg, is_train=False))
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gallery_feats = engine.extract(gallery_loader)  # (G, D)
    print(f"  Gallery: {len(gallery)} 张")

    # 3. 为每张 gallery 找最近训练身份
    print("[3/4] 计算相似度并分配伪标签...")
    sim = torch.matmul(gallery_feats, train_feats.t())  # (G, N)
    max_sim, max_idx = sim.max(dim=1)  # (G,), (G,)
    max_sim = max_sim.cpu().numpy()
    max_idx = max_idx.cpu().numpy()

    pseudo_count = 0
    pseudo_rows = []
    # task.json 中 gallery 的 image_path 相对 task.json 所在目录
    task_dir = Path(args.task_json).parent
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for gi, g in enumerate(gallery):
        conf = float(max_sim[gi])
        if conf < args.threshold:
            continue
        train_idx = int(max_idx[gi])
        ship_id = id2ship[int(train_labels[train_idx])]
        modality = g["modality"]
        # image_path 需要相对输出 CSV 所在目录
        abs_path = (task_dir / g["image_path"]).resolve()
        try:
            rel_path = abs_path.relative_to(out_dir.resolve())
        except ValueError:
            rel_path = abs_path
        pseudo_rows.append({
            "image_id": g["image_id"],
            "ship_id": ship_id,
            "modality": modality,
            "image_path": str(rel_path).replace("\\", "/"),
        })
        pseudo_count += 1
    print(f"  高置信伪标签: {pseudo_count}/{len(gallery)} (阈值={args.threshold})")

    # 4. 合并原始训练 + 伪标签，写出新 CSV
    print("[4/4] 写出合并后的 labels CSV...")
    original_rows = list(csv.DictReader(open(args.train_csv, encoding="utf-8")))
    all_rows = original_rows + pseudo_rows
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "ship_id", "modality", "image_path"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"  原始: {len(original_rows)}, 伪标签: {len(pseudo_rows)}, 总计: {len(all_rows)}")
    print(f"  输出: {args.out_csv}")
    print()
    print("下一步: python train.py --config config/train_gpu.yaml train_labels_csv=" + args.out_csv)


if __name__ == "__main__":
    main()
