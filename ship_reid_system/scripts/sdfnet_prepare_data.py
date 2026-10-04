"""将赛题训练数据转换为 SDF-Net / HOSS 目录结构。

输出结构（SDF-Net 的 datasets/hoss.py 约定）：
    data/HOSS/
    ├── bounding_box_train   # 训练集（labels_train.csv 剔除本地验证集 query/gallery 图像）
    ├── bounding_box_test    # Gallery 集（复用 local_val_task.json 的 gallery）
    └── query                # Query 集（复用 local_val_task.json 的 queries）

文件名规则（与 HOSS/TransOSS 一致，SDF-Net 解析 pid 取首个下划线前的整数）：
    {pid}_{seq}_{RGB|SAR}.tif
    - pid   : 全局连续整数身份 ID（train/query/gallery 共用同一映射，保证同舰船同名）
    - seq   : split 内该身份该模态的序号
    - RGB   : optical（camid=0）
    - SAR   : sar（camid=1，保持三通道灰度复制形式，不做单通道化）

同时输出 identity_map.json（image_id <-> HOSS 文件名），供推理脚本映射回赛题 ID。

用法：
    python scripts/sdfnet_prepare_data.py \
        --labels_train ../赛题6-初赛/训练数据/labels_train.csv \
        --labels_full ../赛题6-初赛/训练数据/labels.csv \
        --task ../赛题6-初赛/训练数据/local_val_task.json \
        --out_dir ../ship_reid_system/SDF-Net/data/HOSS

说明：本地验证集优先复用现有 local_val_task.json（不重新划分），
query/gallery 完全按该文件内容搬运；训练集 = labels_train.csv 剔除验证集图像。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="转换赛题数据为 SDF-Net/HOSS 目录结构")
    p.add_argument("--labels_train", type=str, required=True, help="labels_train.csv（训练子集）")
    p.add_argument("--labels_full", type=str, required=True, help="labels.csv（全量，含验证集身份映射）")
    p.add_argument("--task", type=str, required=True, help="local_val_task.json（复用验证集）")
    p.add_argument("--out_dir", type=str, required=True, help="输出 data/HOSS 目录")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    train_csv = Path(args.labels_train)
    full_csv = Path(args.labels_full)
    task_path = Path(args.task)
    out = Path(args.out_dir)
    data_root = train_csv.parent

    task = json.loads(task_path.read_text(encoding="utf-8"))
    # task.json 的 query 无 image_id 字段，image_id 由 image_path 文件名推断（如 optical/opt_003076.tif -> opt_003076）
    query_ids = [Path(q["image_path"]).stem for q in task["queries"]]
    gallery_ids = [g["image_id"] for g in task["gallery"]]
    q_set, g_set = set(query_ids), set(gallery_ids)
    overlap = q_set & g_set
    print(f"queries: {len(query_ids)}, gallery: {len(gallery_ids)}")
    if overlap:
        print(f"  [注意] query 与 gallery 存在 {len(overlap)} 个同 image_id（推理时需排除 query 自身）")

    # 全量身份映射：labels.csv（7422 行 = 5940 训练 + 1482 验证）覆盖全部图像
    id2ship = {r["image_id"]: r["ship_id"] for r in load_rows(full_csv)}
    # 训练行：local_val_task.json 的 query/gallery 来自初赛公开评测集，不在 labels_train.csv 中，
    # 因此训练集 = labels_train.csv 全部 5940 行（labels.csv 为其超集，身份映射完整）
    train_rows = [r for r in load_rows(train_csv) if r["image_id"] not in q_set and r["image_id"] not in g_set]
    print(f"训练行: {len(load_rows(train_csv))} -> 剔除验证集图像后 {len(train_rows)}")

    # 全局 pid：所有 split 身份统一映射（同 ship_id 同 pid）
    all_ship_ids = sorted({sid for sid in id2ship.values()})
    ship2pid = {sid: i for i, sid in enumerate(all_ship_ids)}
    print(f"身份数: {len(all_ship_ids)}（pid 0..{len(all_ship_ids)-1}）")

    # query 模态从 task image_path 解析
    q_mods = {}
    for q in task["queries"]:
        p = q["image_path"].replace("\\", "/")
        q_mods[Path(q["image_path"]).stem] = "optical" if p.startswith("optical") else "sar"
    items = []
    for r in train_rows:
        items.append((r["image_id"], "bounding_box_train", r["modality"]))
    for iid in query_ids:
        items.append((iid, "query", q_mods[iid]))
    for g in task["gallery"]:
        items.append((g["image_id"], "bounding_box_test", g["modality"]))

    # 分目录写入
    split_dir = {s: out / s for s in ("bounding_box_train", "bounding_box_test", "query")}
    for d in split_dir.values():
        d.mkdir(parents=True, exist_ok=True)
    counter = defaultdict(lambda: {"optical": 0, "sar": 0})
    id_map: dict[str, dict] = {}
    copied = Counter()
    for iid, split, mod in items:
        sid = id2ship.get(iid)
        if sid is None:
            print(f"  [警告] {iid} 无身份来源，跳过")
            continue
        pid = ship2pid[sid]
        seq = counter[pid][mod]
        counter[pid][mod] += 1
        suffix = "RGB" if mod == "optical" else "SAR"
        fname = f"{pid}_{seq}_{suffix}.tif"
        src = data_root / (f"optical/{iid}.tif" if mod == "optical" else f"sar/{iid}.tif")
        dst = split_dir[split] / fname
        if not src.exists():
            print(f"  [警告] 源图不存在: {src}")
            continue
        shutil.copy2(src, dst)
        copied[split] += 1
        id_map[iid] = {"hoss_file": fname, "modality": mod, "pid": pid, "split": split, "ship_id": sid}

    (out / "identity_map.json").write_text(
        json.dumps({"queries": query_ids, "gallery": gallery_ids, "identity_map": id_map},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入完成: {out}")
    for s in ("bounding_box_train", "bounding_box_test", "query"):
        print(f"  {s}: {copied[s]}")
    print(f"  identity_map.json: {out / 'identity_map.json'}")


if __name__ == "__main__":
    main()
