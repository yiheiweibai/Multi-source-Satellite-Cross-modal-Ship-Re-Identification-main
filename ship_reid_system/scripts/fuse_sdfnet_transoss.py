"""SDF-Net 与 TransOSS 跨模型分数融合脚本（本地验证协议评测）。

输入：两份 sim.pt + meta.json（分别由 scripts/sdfnet_inference.py 与
      scripts/transoss_inference.py 的 --save_sim 导出，须基于同一份 task.json）。

融合方法：
  1) weighted : 加权平均。每个模型先按 query×目标候选子集做 min-max 归一化，
                再以 weight_a * sim_a + (1-weight_a) * sim_b 融合。
  2) rrf      : Reciprocal Rank Fusion。每个模型在目标候选子集内独立排序，
                得分 = Σ 1/(60 + rank)，天然不受分数尺度影响。

协议与两个推理脚本一致：按 query_type 过滤候选模态（O2S→SAR、S2O→Optical、
O2O→Optical 且排除 query 自身图像），每 query 取 top10，调用 evaluate.py
--submission 评分。同时输出两个单模型在同一协议下的成绩作对比基线。

用法（在 ship_reid_system 根目录）：
    python scripts/fuse_sdfnet_transoss.py \
        --sim_a sims/sdfnet/sim.pt  --meta_a sims/sdfnet/meta.json \
        --sim_b sims/transoss/sim.pt --meta_b sims/transoss/meta.json \
        --task_json ../赛题6-初赛/训练数据/local_val_task.json \
        --gt_json   ../赛题6-初赛/训练数据/local_val_gt.json \
        --out_dir sims/fused
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent          # ship_reid_system/scripts
SHIP_DIR = SCRIPT_DIR.parent                          # ship_reid_system

QTYPE_TO_TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDF-Net + TransOSS 跨模型分数融合")
    p.add_argument("--sim_a", type=str, required=True, help="模型 A sim.pt（SDF-Net）")
    p.add_argument("--meta_a", type=str, required=True, help="模型 A meta.json")
    p.add_argument("--sim_b", type=str, required=True, help="模型 B sim.pt（TransOSS）")
    p.add_argument("--meta_b", type=str, required=True, help="模型 B meta.json")
    p.add_argument("--name_a", type=str, default="SDF-Net")
    p.add_argument("--name_b", type=str, default="TransOSS")
    p.add_argument("--method", type=str, default="both", choices=["weighted", "rrf", "both"])
    p.add_argument("--weight_a", type=float, default=0.5, help="加权平均中模型 A 的权重（0~1）")
    p.add_argument("--task_json", type=str, required=True)
    p.add_argument("--gt_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=str(SHIP_DIR / "sims" / "fused"))
    return p.parse_args()


def load_pair(sim_path: str, meta_path: str):
    sim = torch.load(sim_path, map_location="cpu")   # [Nq, Ng]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return sim, meta


def align(sim, meta, ref_queries, ref_gallery):
    """把 sim 按参考顺序（ref_queries/ref_gallery 的 image_path 序列）重排，确保两模型行/列对齐。"""
    q_paths = [q["image_path"] for q in ref_queries]
    g_paths = [g["image_path"] for g in ref_gallery]
    q_map = {q["image_path"]: i for i, q in enumerate(meta["queries"])}
    g_map = {g["image_path"]: i for i, g in enumerate(meta["gallery"])}
    q_idx = [q_map[p] for p in q_paths]
    g_idx = [g_map[p] for p in g_paths]
    return sim[q_idx][:, g_idx]


def build_prediction(sim, queries, gallery):
    """与推理脚本相同的协议：按 query_type 过滤候选模态，O2O 排除自身，top10。"""
    g_abs = [g["image_path"] for g in gallery]
    q_abs = [q["image_path"] for q in queries]
    prediction = {}
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        if q["query_type"] == "O2O":
            cand = [i for i in cand if g_abs[i] != q_abs[qi]]
        if len(cand) < 10:
            raise RuntimeError(f"query {q['query_id']} 候选仅 {len(cand)} 个，不足 10（赛题协议要求 ≥10）")
        top = sim[qi, torch.tensor(cand)].topk(10).indices.tolist()
        prediction[q["query_id"]] = [gallery[cand[t]]["image_id"] for t in top]
    return prediction


def minmax_rows(sim, queries, gallery):
    """按 query×目标候选子集 min-max 归一化（避免不同模型分数尺度差异）。"""
    out = torch.zeros_like(sim)
    g_mods = [g["modality"] for g in gallery]
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        if q["query_type"] == "O2O":
            cand = [i for i in cand if gallery[i]["image_path"] != q["image_path"]]
        seg = sim[qi, cand]
        lo, hi = seg.min(), seg.max()
        if hi > lo:
            out[qi, cand] = (seg - lo) / (hi - lo)
        else:
            out[qi, cand] = 0.5
    return out


def rrf_rows(sim, queries, gallery, k=60):
    """每个 query 在目标候选子集内排序，RRF 得分 = Σ 1/(k+rank)。"""
    out = torch.zeros_like(sim)
    g_mods = [g["modality"] for g in gallery]
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        if q["query_type"] == "O2O":
            cand = [i for i in cand if gallery[i]["image_path"] != q["image_path"]]
        seg = sim[qi, cand]
        order = torch.argsort(seg, descending=True)
        ranks = torch.empty(len(cand), dtype=torch.float32)
        ranks[order] = torch.arange(1, len(cand) + 1, dtype=torch.float32)
        out[qi, cand] = 1.0 / (k + ranks)
    return out


def run_eval(pred_path: Path, task_json: Path, gt_json: Path) -> tuple[dict, float]:
    cmd = [sys.executable, str(SHIP_DIR / "evaluate.py"), "--submission",
           "--prediction", str(pred_path), "--task", str(task_json), "--gt", str(gt_json)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SHIP_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py 评测失败: {proc.stderr[-2000:]}")
    scores: dict = {}
    for d in ("O2S", "S2O", "O2O"):
        m = re.search(rf"^{d}\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", proc.stdout, re.M)
        if m:
            scores[d] = float(m.group(3))
    m = re.search(r"综合得分:\s*([\d.]+)", proc.stdout)
    overall = float(m.group(1)) if m else None
    return scores, overall


def main() -> None:
    args = parse_args()
    task_json = Path(args.task_json).resolve()
    gt_json = Path(args.gt_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_a, meta_a = load_pair(args.sim_a, args.meta_a)
    sim_b, meta_b = load_pair(args.sim_b, args.meta_b)
    # 以 A 的 queries/gallery 顺序为参考，B 按 image_path 对齐重排
    ref_q, ref_g = meta_a["queries"], meta_a["gallery"]
    sim_a2 = align(sim_a, meta_a, ref_q, ref_g)
    sim_b2 = align(sim_b, meta_b, ref_q, ref_g)
    print(f"aligned sim_a {tuple(sim_a2.shape)}, sim_b {tuple(sim_b2.shape)}")

    # 单模型协议成绩（同一协议基线）
    single = {}
    for name, s in ((args.name_a, sim_a2), (args.name_b, sim_b2)):
        pred = build_prediction(s, ref_q, ref_g)
        p = out_dir / f"single_{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        single[name] = run_eval(p, task_json, gt_json)
        print(f"[single {name}] overall={single[name][1]:.4f}")

    results = {}
    if args.method in ("weighted", "both"):
        wa = minmax_rows(sim_a2, ref_q, ref_g)
        wb = minmax_rows(sim_b2, ref_q, ref_g)
        fused = args.weight_a * wa + (1 - args.weight_a) * wb
        pred = build_prediction(fused, ref_q, ref_g)
        p = out_dir / f"fused_weighted_w{args.weight_a:.2f}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        results["weighted"] = run_eval(p, task_json, gt_json)
        print(f"[fused weighted w_a={args.weight_a:.2f}] overall={results['weighted'][1]:.4f}")

    if args.method in ("rrf", "both"):
        ra = rrf_rows(sim_a2, ref_q, ref_g)
        rb = rrf_rows(sim_b2, ref_q, ref_g)
        fused = ra + rb
        pred = build_prediction(fused, ref_q, ref_g)
        p = out_dir / "fused_rrf.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        results["rrf"] = run_eval(p, task_json, gt_json)
        print(f"[fused rrf] overall={results['rrf'][1]:.4f}")

    # 汇总表
    print("=" * 76)
    print(f"{'方法':<24}{'O2S':<8}{'S2O':<8}{'O2O':<8}{'综合':<8}")
    for name, (scores, overall) in single.items():
        print(f"{name:<24}{scores.get('O2S', 0):<8.4f}{scores.get('S2O', 0):<8.4f}"
              f"{scores.get('O2O', 0):<8.4f}{overall:<8.4f}")
    for name, (scores, overall) in results.items():
        print(f"fused_{name:<18}{scores.get('O2S', 0):<8.4f}{scores.get('S2O', 0):<8.4f}"
              f"{scores.get('O2O', 0):<8.4f}{overall:<8.4f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
