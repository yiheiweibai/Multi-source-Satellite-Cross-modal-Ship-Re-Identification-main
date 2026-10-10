"""将 CycleGAN 生成的增强图像并入 SDF-Net 训练集（bounding_box_train）。

前提：已运行 sdfnet_prepare_data.py 生成基础 HOSS 数据（含 identity_map.json），
      且已运行 cyclegan/generate.py --mode augment 生成 fake 图像。

用法：
    python scripts/integrate_cyclegan.py \
        --hoss_dir ../SDF-Net/data/HOSS \
        --aug_dir ../SDF-Net/data/HOSS_augmented \
        --out_dir ../SDF-Net/data/HOSS_aug

效果：每个身份在真实 1O+1S 基础上，新增 1 张 fake SAR（来自真实 O）和 1 张 fake optical（来自真实 S），
      即每身份 2O+2S，训练数据量翻倍。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import tifffile as tiff  # noqa: F401  确保 tifffile 可用；shutil.copy2 直接复制二进制


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hoss_dir", type=str, required=True, help="基础 HOSS 目录（含 identity_map.json）")
    p.add_argument("--aug_dir", type=str, required=True, help="generate.py 输出的增强目录")
    p.add_argument("--out_dir", type=str, required=True, help="增强后 HOSS 输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    hoss = Path(args.hoss_dir)
    aug = Path(args.aug_dir)
    out = Path(args.out_dir)

    # 1. 复制基础 HOSS 到 out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(hoss, out)
    train_dir = out / "bounding_box_train"

    # 2. 读取 identity_map，建立 image_id -> {pid, modality} 映射
    id_map_path = out / "identity_map.json"
    id_data = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map = id_data["identity_map"]

    # image_id -> pid（仅训练集）
    iid2pid = {}
    iid2mod = {}
    for iid, info in id_map.items():
        if info["split"] == "bounding_box_train":
            iid2pid[iid] = info["pid"]
            iid2mod[iid] = info["modality"]

    # 3. 统计每个 pid 当前 seq，确定 fake 图像的新 seq
    # 解析现有文件名 {pid}_{seq}_{RGB|SAR}.tif
    pid_max_seq = defaultdict(lambda: {"optical": -1, "sar": -1})
    for f in train_dir.iterdir():
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        seq = int(parts[1])
        mod = "optical" if parts[2] == "RGB" else "sar"
        pid_max_seq[pid][mod] = max(pid_max_seq[pid][mod], seq)

    # 4. 并入 fake sar（来自真实 optical）
    o2s_dir = aug / "optical_to_sar"
    added = 0
    for f in o2s_dir.iterdir():
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        iid = f.stem
        if iid not in iid2pid:
            continue
        pid = iid2pid[iid]
        new_seq = pid_max_seq[pid]["sar"] + 1
        pid_max_seq[pid]["sar"] = new_seq
        dst = train_dir / f"{pid}_{new_seq}_SAR.tif"
        shutil.copy2(f, dst)
        added += 1
    print(f"并入 fake SAR (O->S): {added}")

    # 5. 并入 fake optical（来自真实 sar）
    s2o_dir = aug / "sar_to_optical"
    added2 = 0
    for f in s2o_dir.iterdir():
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        iid = f.stem
        if iid not in iid2pid:
            continue
        pid = iid2pid[iid]
        new_seq = pid_max_seq[pid]["optical"] + 1
        pid_max_seq[pid]["optical"] = new_seq
        dst = train_dir / f"{pid}_{new_seq}_RGB.tif"
        shutil.copy2(f, dst)
        added2 += 1
    print(f"并入 fake optical (S->O): {added2}")

    # 6. 更新 identity_map，记录增强图像（方便追溯）
    aug_entries = {}
    for f in train_dir.iterdir():
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        seq = int(parts[1])
        mod = "optical" if parts[2] == "RGB" else "sar"
        key = f"aug_{pid}_{seq}_{parts[2]}"
        aug_entries[key] = {"hoss_file": f.name, "modality": mod, "pid": pid,
                            "split": "bounding_box_train", "augmented": True}
    id_data["identity_map"].update(aug_entries)
    id_map_path.write_text(json.dumps(id_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n增强完成: {out}")
    print(f"  bounding_box_train 总图像数: {len(list(train_dir.glob('*.tif')))}")
    print(f"  其中 fake 图像: {added + added2}")


if __name__ == "__main__":
    main()
