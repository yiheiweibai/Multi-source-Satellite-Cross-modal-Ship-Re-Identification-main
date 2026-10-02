"""将赛题训练数据转换为 TransOSS 所需的 bounding_box_train 格式。

TransOSS 命名规则：{pid}_{序号}_{RGB|SAR}.tif
  - pid: 身份 ID（整数，从 0 开始）
  - camid: 0=RGB(optical), 1=SAR

用法：
    python scripts/transoss_prepare_data.py \
        --csv ../赛题6-初赛/训练数据/labels_train.csv \
        --out_dir ../transoss_data/HOSS/bounding_box_train
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="训练 labels.csv")
    p.add_argument("--out_dir", type=str, required=True, help="输出 bounding_box_train 目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.csv).parent
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))

    # ship_id -> 连续整数 pid
    ship_ids = sorted(set(r["ship_id"] for r in rows))
    ship2pid = {sid: i for i, sid in enumerate(ship_ids)}
    print(f"身份数: {len(ship_ids)}, 样本数: {len(rows)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 每个身份内光学/SAR 分别编号
    ship_counter = defaultdict(lambda: {"optical": 0, "sar": 0})
    for r in rows:
        pid = ship2pid[r["ship_id"]]
        mod = r["modality"]  # optical / sar
        seq = ship_counter[r["ship_id"]][mod]
        ship_counter[r["ship_id"]][mod] += 1
        suffix = "RGB" if mod == "optical" else "SAR"
        # 文件名: {pid}_{seq}_{RGB|SAR}.tif
        fname = f"{pid}_{seq}_{suffix}.tif"
        src = root / r["image_path"]
        dst = out / fname
        shutil.copy2(src, dst)

    print(f"已复制到 {out}，共 {len(rows)} 张")
    # 统计
    n_optical = sum(1 for r in rows if r["modality"] == "optical")
    n_sar = sum(1 for r in rows if r["modality"] == "sar")
    print(f"  optical: {n_optical}, SAR: {n_sar}")


if __name__ == "__main__":
    main()
