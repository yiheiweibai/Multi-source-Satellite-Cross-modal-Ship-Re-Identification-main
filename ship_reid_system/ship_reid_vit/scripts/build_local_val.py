"""生成本地验证协议：从训练集划分 20% 身份作为验证集，构造 task.json + gt.json。

用法：
    python scripts/build_local_val.py \
        --csv ../../question6-data/traindata/labels.csv \
        --out_dir ../../question6-data/traindata \
        --val_ratio 0.2 --seed 42

产出：
    labels_train.csv   — 训练身份子集（80%），供 train.py 使用
    local_val_task.json — 验证集检索任务（queries + gallery）
    local_val_gt.json   — 验证集 GT（query_id -> [正确 gallery image_id]）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local validation protocol")
    p.add_argument("--csv", type=str, default="", help="官方 labels.csv 路径；留空则用 --config 的 data_root/labels.csv")
    p.add_argument("--out_dir", type=str, default="", help="输出目录；留空则用 --config 的 data_root")
    p.add_argument("--config", type=str, default="config/default.yaml", help="读取 data_root 的配置")
    p.add_argument("--val_ratio", type=float, default=0.2, help="验证集身份占比")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def _resolve_paths(args) -> tuple[str, str]:
    csv_path, out_dir = args.csv, args.out_dir
    if not csv_path or not out_dir:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfgd = yaml.safe_load(f) or {}
        root = cfgd.get("data_root", "data")
        csv_path = csv_path or str(Path(root) / "labels.csv")
        out_dir = out_dir or root
    return csv_path, out_dir


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    csv_path, out_dir_str = _resolve_paths(args)
    print(f"labels.csv: {csv_path} | out_dir: {out_dir_str}")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    # 按 ship_id 分组
    by_ship: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ship[r["ship_id"]].append(r)

    ship_ids = list(by_ship.keys())
    random.shuffle(ship_ids)
    n_val = int(len(ship_ids) * args.val_ratio)
    val_ids = set(ship_ids[:n_val])
    train_ids = set(ship_ids[n_val:])
    print(f"总身份: {len(ship_ids)}, 训练: {len(train_ids)}, 验证: {len(val_ids)}")

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 写出 labels_train.csv（训练身份子集）----
    train_rows = [r for r in rows if r["ship_id"] in train_ids]
    train_csv = out_dir / "labels_train.csv"
    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "ship_id", "modality", "image_path"])
        w.writeheader()
        w.writerows(train_rows)
    print(f"labels_train.csv: {len(train_rows)} 行 -> {train_csv}")

    # ---- 2. 构造验证集 gallery（所有验证身份的图像）----
    val_rows = [r for r in rows if r["ship_id"] in val_ids]
    gallery = []
    # image_id 在验证集内全局唯一
    for r in val_rows:
        gallery.append({
            "image_id": r["image_id"],
            "image_path": r["image_path"],
            "modality": r["modality"],
        })

    # 按 ship_id 分模态
    ship_optical: dict[str, list[str]] = defaultdict(list)   # ship_id -> [image_id]
    ship_sar: dict[str, list[str]] = defaultdict(list)
    for r in val_rows:
        if r["modality"] == "optical":
            ship_optical[r["ship_id"]].append(r["image_id"])
        else:
            ship_sar[r["ship_id"]].append(r["image_id"])

    # ---- 3. 构造 queries + gt ----
    queries = []
    gt: dict[str, list[str]] = {}
    q_counter = 0

    def _make_query(image_id: str, image_path: str, qtype: str, positives: list[str]) -> None:
        nonlocal q_counter
        q_counter += 1
        qid = f"q_{q_counter:06d}"
        queries.append({
            "query_id": qid,
            "image_path": image_path,
            "query_type": qtype,
        })
        gt[qid] = positives

    # image_id -> (image_path, modality, ship_id)
    id2info = {r["image_id"]: r for r in val_rows}

    for sid in val_ids:
        opt = ship_optical.get(sid, [])
        sar = ship_sar.get(sid, [])
        # O2S: optical query, SAR gallery of same ship
        if opt and sar:
            _make_query(opt[0], id2info[opt[0]]["image_path"], "O2S", list(sar))
        # S2O: SAR query, optical gallery of same ship
        if sar and opt:
            _make_query(sar[0], id2info[sar[0]]["image_path"], "S2O", list(opt))
        # O2O: optical query, optical gallery of same ship (需要 >=2 张光学)
        if len(opt) >= 2:
            # 第一张作 query，其余作正样本
            _make_query(opt[0], id2info[opt[0]]["image_path"], "O2O", list(opt[1:]))

    # ---- 4. 写出 local_val_task.json ----
    task = {"queries": queries, "gallery": gallery}
    task_path = out_dir / "local_val_task.json"
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False, indent=2)
    print(f"local_val_task.json: {len(queries)} queries, {len(gallery)} gallery -> {task_path}")

    # ---- 5. 写出 local_val_gt.json ----
    gt_path = out_dir / "local_val_gt.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"local_val_gt.json: {len(gt)} queries -> {gt_path}")

    # 统计
    from collections import Counter
    qt_dist = Counter(q["query_type"] for q in queries)
    print(f"Query 类型分布: {dict(qt_dist)}")


if __name__ == "__main__":
    main()
