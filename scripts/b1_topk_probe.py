# -*- coding: utf-8 -*-
"""B1 深 top-K 探针。

问题：RRF 融合成员当前只暴露 top-10，k=10 的 RRF 数学上不可能抬高召回上限
（只在各成员 top-10 内做重排）。B1 检验：把成员输出扩到 top-K（K=20/30/50/100）
再做 RRF，Final 能否提升；并给出召回上限（union top-K 命中率）的解析。

数据来源
  sims/b1/sim_<tag>/{sim.pt,meta.json}   SDF-Net 成员完整 q x g 相似度矩阵
  sims/b1/shipvit_top100.json            shipvit top-100（--tta --rerank）

用法（工程根目录）：
  .venv-sdfnet\\Scripts\\python.exe scripts\\b1_topk_probe.py
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
KS = (10, 20, 30, 50, 100)

# 成员集：与 reproduce.yaml 的 presets 对照
PRESETS = {
    "ep80": ["ep80"],
    "A_plus_sv": ["ep80", "shipvit"],
    "A_sv_e45_e25": ["ep80", "ep45", "ep25", "shipvit"],
    "final_mos": ["mos75", "ep45", "ep25", "shipvit"],
    "five": ["mos75", "ep80", "ep45", "ep25", "shipvit"],
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------- 赛题指标（复刻 utils/metrics.py） ---------------------------- #
def map_at_k(hit_flags, num_rel, k=10) -> float:
    num_rel = int(num_rel)
    if num_rel <= 0:
        return 0.0
    k = min(int(k), len(hit_flags))
    if k <= 0:
        return 0.0
    hits, ap = 0, 0.0
    for pos in range(k):
        if hit_flags[pos]:
            hits += 1
            ap += hits / (pos + 1)
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
    return {"count": len(sub), "R@1": r1, "mAP@10": mAP10, "score": (r1 + mAP10) / 2}


def evaluate(pred, gt, queries, topk=10):
    res = {d: eval_direction(pred, gt, queries, d, topk) for d in DIRECTIONS}
    res["overall"] = sum(res[d]["score"] * WEIGHTS[d] for d in DIRECTIONS)
    return res


def fold_split(queries):
    """按 query_type 内部交替切分 folds[i % 2]（与 RRF集成方案.md 一致）。"""
    folds = {0: [], 1: []}
    for d in DIRECTIONS:
        sub = [q for q in queries if q["query_type"] == d]
        for i, q in enumerate(sub):
            folds[i % 2].append(q)
    return folds


# ---------------------------- RRF ---------------------------- #
def rrf(member_lists, weights, k=10.0, topk=10):
    out = {}
    for qid in member_lists[0]:
        s = {}
        for lst, w in zip(member_lists, weights):
            for rank, iid in enumerate(lst.get(qid, [])[:], 1):
                s[iid] = s.get(iid, 0.0) + w / (k + rank)
        out[qid] = [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:topk]]
    return out


# ---------------------------- 成员排名构建 ---------------------------- #
def ranked_from_sim(sim_dir: Path, queries, gallery, meta_check=True):
    """由 SDF-Net 导出的 sim.pt 还原每个 query 的完整降序 gallery 排名。"""
    sim = torch.load(sim_dir / "sim.pt", map_location="cpu")
    meta = load_json(sim_dir / "meta.json")
    if meta_check:
        assert [q["query_id"] for q in meta["queries"]] == [q["query_id"] for q in queries], \
            f"{sim_dir}: meta.queries 顺序与 task 不一致"
        assert [g["image_id"] for g in meta["gallery"]] == [g["image_id"] for g in gallery], \
            f"{sim_dir}: meta.gallery 顺序与 task 不一致"
    out = {}
    for qi, q in enumerate(queries):
        tgt = TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == tgt]
        cand = [i for i in cand if gallery[i]["image_path"] != q["image_path"]]
        if not cand:
            out[q["query_id"]] = []
            continue
        vals = sim[qi, torch.tensor(cand, dtype=torch.long)]
        order = torch.argsort(vals, descending=True).tolist()
        out[q["query_id"]] = [gallery[cand[t]]["image_id"] for t in order]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(ROOT.parent / "question6-data" / "traindata" / "local_val_task.json"))
    ap.add_argument("--gt", default=str(ROOT.parent / "question6-data" / "traindata" / "local_val_gt.json"))
    ap.add_argument("--b1_dir", default=str(ROOT / "sims" / "b1"))
    ap.add_argument("--shipvit", default=str(ROOT / "sims" / "b1" / "shipvit_top100.json"))
    ap.add_argument("--dump_topk", type=int, default=0,
                    help=">0 时把各成员的完整排名截断为 top-N 落盘（供官方链路复核）")
    ap.add_argument("--dump_dir", default=str(ROOT / "sims" / "b1"))
    args = ap.parse_args()

    b1 = Path(args.b1_dir)
    task = load_json(args.task)
    queries, gallery = task["queries"], task["gallery"]
    gt = {k: set(v) for k, v in load_json(args.gt).items()}
    folds = fold_split(queries)
    print(f"queries={len(queries)} gallery={len(gallery)} fold0={len(folds[0])} fold1={len(folds[1])}")

    # ---- 成员全排名 ----
    members = {}
    for tag in ("ep80", "ep45", "ep25", "mos75"):
        d = b1 / f"sim_{tag}"
        if not d.exists():
            print(f"[跳过] 缺 {d}")
            continue
        members[tag] = ranked_from_sim(d, queries, gallery)
    if Path(args.shipvit).exists():
        members["shipvit"] = load_json(args.shipvit)
    print("成员就绪:", list(members))

    # ---- 物化成员 top-K 列表（供官方 rrf_fuse.py + evaluate.py 链路复核） ----
    if args.dump_topk > 0:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for tag, rk in members.items():
            p = dump_dir / f"top{args.dump_topk}_{tag}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump({q: rk[q][: args.dump_topk] for q in rk}, f, ensure_ascii=False)
            print(f"[dump] {p}  ({len(rk)} query x top-{args.dump_topk})")

    # ---- 校验：各成员 top-10 单模得分应与其官方预测一致 ----
    print("\n[校验] 成员 top-10 单模得分（应复现官方数值）")
    for tag, rk in members.items():
        r = evaluate({q: rk[q][:10] for q in rk}, gt, queries)
        print(f"  {tag:<9} Final {r['overall']:.4f}  "
              + "  ".join(f"{d} {r[d]['score']:.4f}" for d in DIRECTIONS))

    # ---- fold 基线（锚点） ----
    anchor = members.get("ep80")
    if anchor:
        print("\n[基线] ep80 单模 fold："
              + f"fold0 {evaluate({q['query_id']: anchor[q['query_id']][:10] for q in folds[0]}, gt, folds[0])['overall']:.4f}"
              + f" / fold1 {evaluate({q['query_id']: anchor[q['query_id']][:10] for q in folds[1]}, gt, folds[1])['overall']:.4f}")

    # ---- 召回上限分析 ----
    print("\n[召回上限] union 成员 top-K 命中率（GT 落在任一成员 top-K 内的 query 占比）")
    for name, tags in PRESETS.items():
        tags = [t for t in tags if t in members]
        if not tags:
            continue
        row = []
        for K in KS:
            hit_any, hit_deep = 0, 0
            for q in queries:
                qid = q["query_id"]
                pos = gt.get(qid, set())
                u10 = set()
                uK = set()
                for t in tags:
                    lst = members[t].get(qid, [])
                    u10.update(lst[:10])
                    uK.update(lst[:K])
                if pos & u10:
                    hit_any += 1
                elif pos & uK:
                    hit_deep += 1
            n = len(queries)
            row.append(f"K={K}: {(hit_any + hit_deep) / n:.4f} (top10 {hit_any / n:.4f} + 深挖 {hit_deep / n:.4f})")
        print(f"  {name:<15} " + " | ".join(row))

    # ---- 主结果：不同 K 下的 RRF ----
    print("\n[主结果] RRF(等权, k=10) 在不同成员截断 K 下的 Final / fold0 / fold1")
    hdr = f"  {'preset':<15}{'K':>5}{'Final':>10}{'fold0':>10}{'fold1':>10}   " + "  ".join(f"{d:>7}" for d in DIRECTIONS)
    print(hdr)
    for name, tags in PRESETS.items():
        tags = [t for t in tags if t in members]
        if not tags:
            continue
        for K in KS:
            ml = [{q: members[t].get(q, [])[:K] for q in members[t]} for t in tags]
            fused = rrf(ml, [1.0] * len(tags), k=10.0, topk=10)
            full = evaluate(fused, gt, queries)
            f0 = evaluate({q["query_id"]: fused[q["query_id"]] for q in folds[0]}, gt, folds[0])["overall"]
            f1 = evaluate({q["query_id"]: fused[q["query_id"]] for q in folds[1]}, gt, folds[1])["overall"]
            line = (f"  {name:<15}{K:>5}{full['overall']:>10.4f}{f0:>10.4f}{f1:>10.4f}   "
                    + "  ".join(f"{full[d]['score']:>7.4f}" for d in DIRECTIONS))
            print(line)

    # ---- 参考：深 K 下扫描 RRF 常数 k（重点看两折是否同时为正） ----
    print("\n[参考] K=100 下扫描 RRF 常数 k（每格：Final / fold0 / fold1；基线 = 该 preset 的 K=10,k=10）")
    ol = {t: members[t] for t in members}
    for name, tags in PRESETS.items():
        tags = [t for t in tags if t in members]
        if len(tags) < 2:
            continue
        ml = [{q: ol[t].get(q, [])[:100] for q in ol[t]} for t in tags]
        base = evaluate(rrf([{q: ol[t].get(q, [])[:10] for q in ol[t]} for t in tags],
                            [1.0] * len(tags), k=10.0, topk=10), gt, queries)["overall"]
        b0 = evaluate({q["query_id"]: rrf([{q2: ol[t].get(q2, [])[:10] for q2 in folds[0]} for t in tags],
                                          [1.0] * len(tags), k=10.0, topk=10)[q["query_id"]]
                       for q in folds[0]}, gt, folds[0])["overall"] if False else None
        # 基线两折（直接复用 K=10 的融合）
        f10 = rrf([{q: ol[t].get(q, [])[:10] for q in ol[t]} for t in tags], [1.0] * len(tags), k=10.0, topk=10)
        b0 = evaluate({q["query_id"]: f10[q["query_id"]] for q in folds[0]}, gt, folds[0])["overall"]
        b1 = evaluate({q["query_id"]: f10[q["query_id"]] for q in folds[1]}, gt, folds[1])["overall"]
        print(f"  -- {name}  基线 K=10/k=10: {base:.4f} (fold0 {b0:.4f} fold1 {b1:.4f})")
        for kk in (1, 2, 3, 5, 8, 10, 15, 20):
            fused = rrf(ml, [1.0] * len(tags), k=float(kk), topk=10)
            full = evaluate(fused, gt, queries)
            f0 = evaluate({q["query_id"]: fused[q["query_id"]] for q in folds[0]}, gt, folds[0])["overall"]
            f1 = evaluate({q["query_id"]: fused[q["query_id"]] for q in folds[1]}, gt, folds[1])["overall"]
            ok = "OK " if (f0 - b0 >= 0.003 and f1 - b1 >= 0.003) else "   "
            print(f"     {ok}k={kk:<3} {full['overall']:.4f} ({full['overall'] - base:+.4f})  "
                  f"fold0 {f0:.4f} ({f0 - b0:+.4f})  fold1 {f1:.4f} ({f1 - b1:+.4f})")


if __name__ == "__main__":
    main()