"""生成 SDF-Net 全量训练数据目录（波次5：labels.csv 全量 3534 身份/7422 图重训）。

与 sdfnet_prepare_data.py 的区别：
  - 训练目录 = labels.csv 全部图像（不剔除本地验证集身份）；
  - query / bounding_box_test 不重新搬运，改用 junction 链接到现役 data/HOSS/ 下
    同名目录（内置 EVAL 仅作训练日志，不用于选模——铁律）；
  - 不生成 identity_map.json（测试集推理走官方 task.json，与训练目录无关）。

目录结构（hoss.py 约定 dataset_dir = ROOT_DIR/HOSS）：
    SDF-Net/data/full/
    └── HOSS/
        ├── bounding_box_train   # labels.csv 全量 7422 图（hardlink，不占额外磁盘）
        ├── bounding_box_test    # junction -> ../../HOSS/bounding_box_test
        └── query                # junction -> ../../HOSS/query

文件名规则与 sdfnet_prepare_data.py 完全一致：{pid}_{seq}_{RGB|SAR}.tif，
pid = sorted(ship_ids) 的全局映射。

用法：
    python scripts/sdfnet_prepare_full.py --labels <traindata>/labels.csv --out_dir SDF-Net/data/full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 SDF-Net 全量训练目录（HOSS 结构）")
    p.add_argument("--labels", type=str, required=True, help="官方全量 labels.csv")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录（其下将创建 HOSS/）")
    p.add_argument("--ref_hoss", type=str, default="", help="现役 data/HOSS 目录（query/gallery junction 来源，默认取 out_dir 同级 HOSS）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = Path(args.labels)
    out_root = Path(args.out_dir)
    hoss_dir = out_root / "HOSS"
    train_dir = hoss_dir / "bounding_box_train"
    # junction 来源：显式指定，或默认 SDF-Net/data/HOSS（out_dir 的上两级）
    ref = Path(args.ref_hoss) if args.ref_hoss else hoss_dir.parent.parent / "HOSS"

    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    ship_ids = sorted({r["ship_id"] for r in rows})
    ship2pid = {sid: i for i, sid in enumerate(ship_ids)}
    print(f"labels.csv: {len(rows)} 图 / {len(ship_ids)} 身份（pid 0..{len(ship_ids)-1}）")

    train_dir.mkdir(parents=True, exist_ok=True)

    # 训练目录：labels.csv 全量，hardlink 源图（同卷，省磁盘与时间）
    counter = defaultdict(lambda: {"optical": 0, "sar": 0})
    made = 0
    missing = 0
    for r in rows:
        iid, mod, sid = r["image_id"], r["modality"], r["ship_id"]
        pid = ship2pid[sid]
        seq = counter[pid][mod]
        counter[pid][mod] += 1
        suffix = "RGB" if mod == "optical" else "SAR"
        dst = train_dir / f"{pid}_{seq}_{suffix}.tif"
        src = labels.parent / (f"optical/{iid}.tif" if mod == "optical" else f"sar/{iid}.tif")
        if not src.exists():
            missing += 1
            print(f"  [警告] 源图不存在: {src}")
            continue
        if dst.exists():
            made += 1
            continue
        os.link(src, dst)
        made += 1
    print(f"bounding_box_train: {made} 图（hardlink），缺失 {missing}")

    # query / bounding_box_test：junction 复用现役目录（内置 EVAL 日志用，不参与选模）
    for sub in ("query", "bounding_box_test"):
        dst = hoss_dir / sub
        src = ref / sub
        if dst.exists():
            print(f"{sub}: 已存在，跳过")
            continue
        if not src.exists():
            raise RuntimeError(f"junction 源不存在: {src}")
        os.system(f'mklink /J "{dst}" "{src}"')
        print(f"{sub}: junction -> {src}")

    # 冒烟统计：每身份模态分布
    both = sum(1 for p in counter if counter[p]["optical"] > 0 and counter[p]["sar"] > 0)
    print(f"双模态身份数（RGB-SAR pair 数）: {both}")
    (out_root / "prepare_full_summary.json").write_text(json.dumps(
        {"labels": str(labels), "n_images": len(rows), "n_ids": len(ship_ids),
         "n_dual_modal_ids": both, "pid_map_head": {k: ship2pid[k] for k in ship_ids[:5]}},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
