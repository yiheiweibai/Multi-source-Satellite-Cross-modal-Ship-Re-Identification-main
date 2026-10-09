"""多次随机划分验证：生成 N 个不同 seed 的划分，分别推理+评测，取均值±标准差。

用法（需训练好的 checkpoint）：
    python scripts/multi_split_val.py \
        --ckpt outputs/checkpoints/best.pth \
        --csv ../../question6-data/traindata/labels.csv \
        --out_dir ../../question6-data/traindata \
        --num_splits 5 --val_ratio 0.2

仅生成分划（不推理）：加 --gen_only
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-split validation")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint 路径（gen_only 时可空）")
    p.add_argument("--csv", type=str, required=True, help="官方 labels.csv")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")
    p.add_argument("--num_splits", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--gen_only", action="store_true", help="仅生成分划，不推理评测")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--qe", action="store_true")
    p.add_argument("--cluster", action="store_true")
    return p.parse_args()


def gen_split(csv_path: str, out_dir: Path, seed: int, val_ratio: float) -> tuple[Path, Path]:
    """生成单个划分，返回 (task_path, gt_path)。"""
    import csv as csv_mod
    rows = list(csv_mod.DictReader(open(csv_path, encoding="utf-8")))
    by_ship = defaultdict(list)
    for r in rows:
        by_ship[r["ship_id"]].append(r)
    ship_ids = list(by_ship.keys())
    random.Random(seed).shuffle(ship_ids)
    n_val = int(len(ship_ids) * val_ratio)
    val_ids = set(ship_ids[:n_val])
    val_rows = [r for r in rows if r["ship_id"] in val_ids]

    gallery = [{"image_id": r["image_id"], "image_path": r["image_path"], "modality": r["modality"]} for r in val_rows]
    ship_optical = defaultdict(list)
    ship_sar = defaultdict(list)
    for r in val_rows:
        if r["modality"] == "optical":
            ship_optical[r["ship_id"]].append(r["image_id"])
        else:
            ship_sar[r["ship_id"]].append(r["image_id"])
    id2info = {r["image_id"]: r for r in val_rows}

    queries, gt = [], {}
    qn = 0
    for sid in val_ids:
        opt, sar = ship_optical.get(sid, []), ship_sar.get(sid, [])
        if opt and sar:
            qn += 1
            qid = f"q_{qn:06d}"
            queries.append({"query_id": qid, "image_path": id2info[opt[0]]["image_path"], "query_type": "O2S"})
            gt[qid] = list(sar)
        if sar and opt:
            qn += 1
            qid = f"q_{qn:06d}"
            queries.append({"query_id": qid, "image_path": id2info[sar[0]]["image_path"], "query_type": "S2O"})
            gt[qid] = list(opt)
        if len(opt) >= 2:
            qn += 1
            qid = f"q_{qn:06d}"
            queries.append({"query_id": qid, "image_path": id2info[opt[0]]["image_path"], "query_type": "O2O"})
            gt[qid] = list(opt[1:])

    task_path = out_dir / f"local_val_task_s{seed}.json"
    gt_path = out_dir / f"local_val_gt_s{seed}.json"
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump({"queries": queries, "gallery": gallery}, f, ensure_ascii=False, indent=2)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    return task_path, gt_path


def run_eval(ckpt: str, task_path: Path, gt_path: Path, post_flags: list[str]) -> dict:
    """对单个划分推理+评测，返回各方向指标。"""
    repo = Path(__file__).resolve().parent.parent
    pred_path = task_path.parent / f"pred_{task_path.stem}.json"
    py = repo / ".venv" / "Scripts" / "python.exe"
    cmd_inf = [
        str(py), "inference.py", "--config", "config/default.yaml",
        "--ckpt", ckpt, "--task_json", str(task_path), "--out_prediction", str(pred_path),
    ] + post_flags
    print(f"  [推理] seed={task_path.stem}")
    r = subprocess.run(cmd_inf, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-500:], r.stderr[-500:])
        return {}
    cmd_eval = [
        str(py), "evaluate.py", "--submission",
        "--prediction", str(pred_path), "--task", str(task_path), "--gt", str(gt_path),
    ]
    r = subprocess.run(cmd_eval, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-500:], r.stderr[-500:])
        return {}
    # 解析评测输出
    result = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in ("O2S", "S2O", "O2O"):
            result[parts[0]] = {"R@1": float(parts[2]), "mAP@10": float(parts[3]), "score": float(parts[4])}
        elif "综合得分" in line:
            result["overall"] = float(line.split(":")[1].strip())
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    post_flags = []
    for f in ("tta", "rerank", "qe", "cluster"):
        if getattr(args, f):
            post_flags.append(f"--{f}")

    all_results = []
    for i in range(args.num_splits):
        seed = args.base_seed + i
        task_path, gt_path = gen_split(args.csv, out_dir, seed, args.val_ratio)
        print(f"[划分 {i+1}/{args.num_splits}] seed={seed}")
        if args.gen_only:
            continue
        res = run_eval(args.ckpt, task_path, gt_path, post_flags)
        if res:
            all_results.append(res)
            for d in ("O2S", "S2O", "O2O"):
                if d in res:
                    print(f"    {d}: R@1={res[d]['R@1']:.4f} mAP={res[d]['mAP@10']:.4f} score={res[d]['score']:.4f}")
            if "overall" in res:
                print(f"    综合: {res['overall']:.4f}")

    if args.gen_only or not all_results:
        return

    # 聚合
    print("\n" + "=" * 60)
    print(f"多次划分结果（{len(all_results)} 次）")
    print("=" * 60)
    import statistics
    for d in ("O2S", "S2O", "O2O", "overall"):
        if d == "overall":
            vals = [r.get("overall", 0) for r in all_results]
        else:
            vals = [r.get(d, {}).get("score", 0) for r in all_results]
        if vals:
            print(f"{d:<8} 综合得分: {statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
