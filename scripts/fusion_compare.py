# -*- coding: utf-8 -*-
"""融合方法横评（无泄漏 2828 local_val 口径）

对比方法：
  A. RRF k-sweep（k=1,2,3,5,7,10, member_topk=100）—— 现役基线
  B. Borda Count（线性排名加权，member_topk=100）
  C. 分数融合 z-score（每 query 每成员 z-score 归一化后加权平均）
  D. 分数融合 min-max（每 query 每成员 min-max 归一化后加权平均）
  E. 混合：RRF k=3 top-50 候选池 → z-score 分数重排 top-10

用法：
  python scripts/fusion_compare.py
  python scripts/fusion_compare.py --verbose  # 逐方法逐方向打印
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DIRECTIONS = ("O2S", "S2O", "O2O")
WEIGHTS = {"O2S": 0.45, "S2O": 0.45, "O2O": 0.10}
TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}

# ---- 成员 sim 矩阵路径 + 权重（final 2828 预设 + DINOv2-B ep040/060）----
MEMBERS = {
    "sdfnet_mos75":        ("sims/localval/sdfsim_mos75/sim.pt", 1.0),
    "sdfnet_ep45":         ("sims/localval/sdfsim_ep45/sim.pt",  1.0),
    "sdfnet_ep25":         ("sims/localval/sdfsim_ep25/sim.pt",  1.0),
    "shipvit":             ("sims/localval/sim_shipvit.pt",             1.0),
    "convnext_t_ep080":    ("sims/localval/sim_convnext_t_ep080.pt",     0.5),
    "convnext_s_ep060":    ("sims/localval/sim_convnext_s_ep060.pt",     0.5),
    "convnext_s_pure_ep040": ("sims/localval/sim_convnext_s_pure_ep040.pt", 0.75),
    "swin_s_v2_ep060":     ("sims/localval/sim_swin_s_v2_ep060.pt",      1.0),
    "dinov2_b_ep040":      ("sims/screen_dinov2_b/sim_dinov2_b_ep040.pt", 1.0),
    "dinov2_b_ep060":      ("sims/screen_dinov2_b/sim_dinov2_b_ep060.pt", 1.0),
}

TASK = ROOT.parent / "question6-data" / "traindata" / "local_val_task.json"
GT   = ROOT.parent / "question6-data" / "traindata" / "local_val_gt.json"


def load_task():
    with open(TASK, encoding="utf-8") as f:
        return json.load(f)
    with open(GT, encoding="utf-8") as f:
        return json.load(f)


def load_gt():
    with open(GT, encoding="utf-8") as f:
        gt = json.load(f)
    return {k: set(v) if isinstance(v, list) else set([v]) for k, v in gt.items()}


def map_at_k(hits, num_rel, k=10):
    if num_rel == 0:
        return 0.0
    ap = 0.0
    hits_found = 0
    for i, h in enumerate(hits[:k], 1):
        if h:
            hits_found += 1
            ap += hits_found / i
    return ap / min(num_rel, k)


def eval_direction(pred, gt, queries, direction, topk=10):
    sub = [q for q in queries if q["query_type"] == direction]
    n = max(len(sub), 1)
    r1, ap_sum = 0, 0.0
    for q in sub:
        cands = pred.get(q["query_id"], [])[:topk]
        pos = gt.get(q["query_id"], set())
        hits = [1 if c in pos else 0 for c in cands]
        if hits and hits[0] == 1:
            r1 += 1
        ap_sum += map_at_k(hits, len(pos), k=topk)
    r1, mAP10 = r1 / n, ap_sum / n
    return {"R@1": r1, "mAP@10": mAP10, "score": (r1 + mAP10) / 2}


def evaluate(pred, gt, queries, topk=10):
    res = {d: eval_direction(pred, gt, queries, d, topk) for d in DIRECTIONS}
    res["Final"] = sum(res[d]["score"] * WEIGHTS[d] for d in DIRECTIONS)
    return res


# ---- sim 矩阵预加载 + 模态掩码 ----
def load_sims(queries, gallery):
    sims = {}
    for name, (rel, w) in MEMBERS.items():
        p = ROOT / rel
        sim = torch.load(p, map_location="cpu")
        assert tuple(sim.shape) == (len(queries), len(gallery)), \
            f"{name}: {tuple(sim.shape)} != ({len(queries)},{len(gallery)})"
        sims[name] = sim
    return sims


def mod_mask(queries, gallery):
    """每 query 的目标模态 gallery 索引 + 自身排除。"""
    g_paths = [g["image_path"] for g in gallery]
    by_mod = {mod: [i for i, g in enumerate(gallery) if g["modality"] == mod]
              for mod in ("sar", "optical")}
    masks = []
    for q in queries:
        cand = list(by_mod[TARGET_MOD[q["query_type"]]])
        qp = q["image_path"]
        cand = [i for i in cand if g_paths[i] != qp]
        masks.append(cand)
    return masks


# ---- 融合方法 ----
def fuse_rrf(sims, masks, queries, gallery, weights, k, member_topk=100, topk=10):
    out = {}
    for qi, q in enumerate(queries):
        cand = masks[qi]
        scores = {}
        for name, sim in sims.items():
            w = weights[name]
            vals = sim[qi, torch.tensor(cand, dtype=torch.long)]
            order = torch.argsort(vals, descending=True).tolist()
            for rank, idx in enumerate(order[:member_topk], 1):
                gid = gallery[cand[idx]]["image_id"]
                scores[gid] = scores.get(gid, 0.0) + w / (k + rank)
        out[q["query_id"]] = [gid for gid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def fuse_borda(sims, masks, queries, gallery, weights, member_topk=100, topk=10):
    out = {}
    for qi, q in enumerate(queries):
        cand = masks[qi]
        N = min(member_topk, len(cand))
        scores = {}
        for name, sim in sims.items():
            w = weights[name]
            vals = sim[qi, torch.tensor(cand, dtype=torch.long)]
            order = torch.argsort(vals, descending=True).tolist()
            for rank, idx in enumerate(order[:member_topk], 1):
                gid = gallery[cand[idx]]["image_id"]
                scores[gid] = scores.get(gid, 0.0) + w * (N - rank)
        out[q["query_id"]] = [gid for gid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def fuse_score(sims, masks, queries, gallery, weights, mode="zscore", member_topk=0, topk=10):
    """分数融合：每 query 每成员归一化后加权平均。member_topk=0 表示用全量 gallery。"""
    out = {}
    for qi, q in enumerate(queries):
        cand = masks[qi]
        cand_t = torch.tensor(cand, dtype=torch.long)
        agg = torch.zeros(len(cand))
        for name, sim in sims.items():
            w = weights[name]
            vals = sim[qi, cand_t].float()
            if member_topk > 0 and member_topk < len(cand):
                # 只取该成员 top-member_topk 的分数，其余置 0
                topk_vals, topk_idx = torch.topk(vals, min(member_topk, len(cand)))
                mask = torch.zeros_like(vals)
                mask[topk_idx] = 1.0
                vals = vals * mask
            if mode == "zscore":
                mu, sd = vals.mean(), vals.std() + 1e-8
                vals = (vals - mu) / sd
            elif mode == "minmax":
                mn, mx = vals.min(), vals.max()
                vals = (vals - mn) / (mx - mn + 1e-8)
            elif mode == "softmax":
                vals = torch.softmax(vals, dim=0)
            agg += w * vals
        order = torch.argsort(agg, descending=True).tolist()
        out[q["query_id"]] = [gallery[cand[i]]["image_id"] for i in order[:topk]]
    return out


def fuse_hybrid(sims, masks, queries, gallery, weights, k=3.0, pool=50, topk=10):
    """RRF k=3 取 top-pool 候选池 → z-score 分数重排 → top-10。"""
    out = {}
    for qi, q in enumerate(queries):
        cand = masks[qi]
        cand_t = torch.tensor(cand, dtype=torch.long)
        # Phase 1: RRF
        rrf_scores = {}
        for name, sim in sims.items():
            w = weights[name]
            vals = sim[qi, cand_t]
            order = torch.argsort(vals, descending=True).tolist()
            for rank, idx in enumerate(order[:100], 1):
                gid = gallery[cand[idx]]["image_id"]
                rrf_scores[gid] = rrf_scores.get(gid, 0.0) + w / (k + rank)
        pool_ids = [gid for gid, _ in sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:pool]]
        # Phase 2: z-score rerank within pool
        pool_idx = {gid: i for i, gid in enumerate(pool_ids)}
        agg = torch.zeros(len(pool_ids))
        for name, sim in sims.items():
            w = weights[name]
            vals = sim[qi, cand_t]
            member_scores = {}
            for rank, idx in enumerate(torch.argsort(vals, descending=True).tolist()[:100], 1):
                gid = gallery[cand[idx]]["image_id"]
                if gid in pool_idx:
                    member_scores[gid] = vals[idx].item()
            if member_scores:
                raw = torch.tensor(list(member_scores.values()))
                mu, sd = raw.mean(), raw.std() + 1e-8
                for gid, val in member_scores.items():
                    agg[pool_idx[gid]] += w * ((val - mu) / sd)
        order = torch.argsort(agg, descending=True).tolist()
        out[q["query_id"]] = [pool_ids[i] for i in order[:topk]]
    return out


def fmt(res):
    return (f"Final={res['Final']:.4f} | "
            f"O2S={res['O2S']['score']:.4f} S2O={res['S2O']['score']:.4f} O2O={res['O2O']['score']:.4f}")


def fold_split(queries):
    """按 query_type 内部交替切分 folds[i % 2]（与生产一致）。"""
    folds = {0: [], 1: []}
    for d in DIRECTIONS:
        sub = [q for q in queries if q["query_type"] == d]
        for i, q in enumerate(sub):
            folds[i % 2].append(q)
    return folds


def fold_eval(pred, gt, folds):
    out = []
    for f in (0, 1):
        sub = folds[f]
        sub_pred = {q["query_id"]: pred[q["query_id"]] for q in sub}
        out.append(evaluate(sub_pred, gt, sub)["Final"])
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    task = load_task()
    queries, gallery = task["queries"], task["gallery"]
    gt = load_gt()
    weights = {n: w for n, (_, w) in MEMBERS.items()}
    folds = fold_split(queries)

    print(f"成员数: {len(MEMBERS)} | queries: {len(queries)} | gallery: {len(gallery)}")
    print("加载 sim 矩阵...")
    sims = load_sims(queries, gallery)
    masks = mod_mask(queries, gallery)
    print("加载完成。\n")

    results = {}

    # A. RRF k-sweep
    print("=" * 70)
    print("A. RRF k-sweep (member_topk=100)")
    print("=" * 70)
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
        pred = fuse_rrf(sims, masks, queries, gallery, weights, k=float(k))
        res = evaluate(pred, gt, queries)
        tag = f"RRF k={k}"
        results[tag] = res
        print(f"  {tag:15s}  {fmt(res)}")

    # B. Borda
    print("\n" + "=" * 70)
    print("B. Borda Count (member_topk=100)")
    print("=" * 70)
    pred = fuse_borda(sims, masks, queries, gallery, weights, member_topk=100)
    res = evaluate(pred, gt, queries)
    results["Borda"] = res
    print(f"  {'Borda':15s}  {fmt(res)}")

    # C. Score z-score (full gallery)
    print("\n" + "=" * 70)
    print("C. 分数融合 z-score (全量 gallery)")
    print("=" * 70)
    pred = fuse_score(sims, masks, queries, gallery, weights, mode="zscore")
    res = evaluate(pred, gt, queries)
    results["Score-zscore"] = res
    print(f"  {'Score-zscore':15s}  {fmt(res)}")

    # C2. Score z-score (member_topk=100)
    pred = fuse_score(sims, masks, queries, gallery, weights, mode="zscore", member_topk=100)
    res = evaluate(pred, gt, queries)
    results["Score-zscore-tk100"] = res
    print(f"  {'Score-zs-tk100':15s}  {fmt(res)}")

    # D. Score min-max (full gallery)
    print("\n" + "=" * 70)
    print("D. 分数融合 min-max (全量 gallery)")
    print("=" * 70)
    pred = fuse_score(sims, masks, queries, gallery, weights, mode="minmax")
    res = evaluate(pred, gt, queries)
    results["Score-minmax"] = res
    print(f"  {'Score-minmax':15s}  {fmt(res)}")

    # D2. Score min-max (member_topk=100)
    pred = fuse_score(sims, masks, queries, gallery, weights, mode="minmax", member_topk=100)
    res = evaluate(pred, gt, queries)
    results["Score-minmax-tk100"] = res
    print(f"  {'Score-mm-tk100':15s}  {fmt(res)}")

    # E. Hybrid RRF + z-score rerank
    print("\n" + "=" * 70)
    print("E. 混合 RRF(k=3,top-50) → z-score 重排")
    print("=" * 70)
    for pool in [30, 50, 100]:
        pred = fuse_hybrid(sims, masks, queries, gallery, weights, k=3.0, pool=pool)
        res = evaluate(pred, gt, queries)
        tag = f"Hybrid pool={pool}"
        results[tag] = res
        print(f"  {tag:15s}  {fmt(res)}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总（按 Final 降序）")
    print("=" * 70)
    for tag, res in sorted(results.items(), key=lambda kv: -kv[1]["Final"]):
        print(f"  {tag:20s}  {fmt(res)}")

    # 折叠稳健性：仅对关键方法做 fold0/fold1
    print("\n" + "=" * 70)
    print("折叠稳健性（fold0 / fold1，parity 切分）")
    print("=" * 70)
    key_cases = {
        "RRF k=3": lambda: fuse_rrf(sims, masks, queries, gallery, weights, k=3.0),
        "RRF k=10": lambda: fuse_rrf(sims, masks, queries, gallery, weights, k=10.0),
        "Score-minmax": lambda: fuse_score(sims, masks, queries, gallery, weights, mode="minmax"),
        "Hybrid pool=50": lambda: fuse_hybrid(sims, masks, queries, gallery, weights, k=3.0, pool=50),
        "Hybrid pool=100": lambda: fuse_hybrid(sims, masks, queries, gallery, weights, k=3.0, pool=100),
    }
    for tag, fn in key_cases.items():
        pred = fn()
        f0, f1 = fold_eval(pred, gt, folds)
        print(f"  {tag:18s}  fold0={f0:.4f}  fold1={f1:.4f}  min={min(f0, f1):.4f}")

    # 最佳
    best = max(results.items(), key=lambda kv: kv[1]["Final"])
    baseline = results.get("RRF k=3", None)
    print(f"\n最佳: {best[0]}  {fmt(best[1])}")
    if baseline:
        delta = best[1]["Final"] - baseline["Final"]
        print(f"vs 现役 RRF k=3 ({baseline['Final']:.4f}): {'+' if delta >= 0 else ''}{delta:.4f}")


if __name__ == "__main__":
    main()
