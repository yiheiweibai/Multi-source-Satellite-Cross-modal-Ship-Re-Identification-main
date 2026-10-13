# -*- coding: utf-8 -*-
"""新成员三重门槛筛选（缩水无泄漏口径，本地验证 1450 query）。

铁律（见 project_memory）：
  G1  生产 k=3 上，两折（fold0/fold1）Final 增益同时 >= +0.003；
  G2  k 扫描 {1,3,5,10,20,40} 中，任何 k 的任何一折增益都不得为负（浮点容差 1e-4）；
  G3  人工复核 k 扫描与方向拆解。

指标：方向分 = (R@1 + mAP@10)/2；Final = 0.45*O2S + 0.45*S2O + 0.10*O2O。
折切分：按 query_type 内部交替 folds[i % 2]。
召回口径：query_type -> 目标模态 O2S:sar / S2O:optical / O2O:optical；剔除 query 自身图像。

数据来源（sims/localval，均为 1450x1482 的 k-reciprocal(tta) 相似度矩阵）：
  基线成员 = `--base_preset final` 的 8 个成员（见 PRESETS）；
  候选成员 = `--cand_dir` 下 sim_<tag>.pt（由 inference.py --local_val_task --tta --rerank --save_sim 产出）。

用法（工程根目录；建议用 ship_reid_vit 的 venv，torch 版本匹配）：
  ship_reid_vit\\venv\\Scripts\\python.exe scripts\\member_screen.py ^
      --cand_dir sims\\strong_cm --cand strong_cm_ep020 strong_cm_ep040
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

# 基线成员池：tag -> sims/localval 下的相对路径
BASE_MEMBERS = {
    "mos75": "sdfsim_mos75/sim.pt",
    "ep45": "sdfsim_ep45/sim.pt",
    "ep25": "sdfsim_ep25/sim.pt",
    "shipvit": "sim_shipvit.pt",
    "convnext_t_ep080": "sim_convnext_t_ep080.pt",
    "convnext_s_ep060": "sim_convnext_s_ep060.pt",
    "convnext_s_pure_ep040": "sim_convnext_s_pure_ep040.pt",
    "swin_s_v2_ep060": "sim_swin_s_v2_ep060.pt",
}
# 权重预设（与 configs/reproduce.yaml 的 final 预设逐项一致）
PRESETS = {
    "final": {
        "mos75": 1.0,
        "ep45": 1.0,
        "ep25": 1.0,
        "shipvit": 1.0,
        "convnext_t_ep080": 1.0,
        "convnext_s_ep060": 0.5,
        "convnext_s_pure_ep040": 0.75,
        "swin_s_v2_ep060": 1.0,
    },
}
# 基线锚点（k=3，容差 ANCHOR_TOL）
ANCHORS = {"overall": 0.6905, "fold0": 0.7046, "fold1": 0.6764}
ANCHOR_TOL = 5e-4


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


def fold_eval(fused, gt, folds):
    out = []
    for f in (0, 1):
        sub = folds[f]
        out.append(evaluate({q["query_id"]: fused[q["query_id"]] for q in sub}, gt, sub)["overall"])
    return out[0], out[1]


# ---------------------------- RRF ---------------------------- #
def rrf(member_lists, weights, k=3.0, topk=10):
    out = {}
    for qid in member_lists[0]:
        s = {}
        for lst, w in zip(member_lists, weights):
            for rank, iid in enumerate(lst.get(qid, []), 1):
                s[iid] = s.get(iid, 0.0) + w / (k + rank)
        out[qid] = [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:topk]]
    return out


# ---------------------------- 成员排名构建 ---------------------------- #
def ranked_from_sim(sim_path: Path, queries, gallery, exclude_self=True):
    """由相似度矩阵还原每个 query 的完整降序 gallery 排名（按目标模态掩码）。"""
    sim = torch.load(sim_path, map_location="cpu")
    assert tuple(sim.shape) == (len(queries), len(gallery)), \
        f"{sim_path}: shape {tuple(sim.shape)} != ({len(queries)},{len(gallery)})"
    g_paths = [g["image_path"] for g in gallery]
    by_mod = {
        mod: [i for i, g in enumerate(gallery) if g["modality"] == mod]
        for mod in ("sar", "optical")
    }
    out = {}
    for qi, q in enumerate(queries):
        cand = list(by_mod[TARGET_MOD[q["query_type"]]])
        if exclude_self:
            q_path = q["image_path"]
            cand = [i for i in cand if g_paths[i] != q_path]
        vals = sim[qi, torch.tensor(cand, dtype=torch.long)]
        order = torch.argsort(vals, descending=True).tolist()
        out[q["query_id"]] = [gallery[cand[t]]["image_id"] for t in order]
    return out


def build_rankings(tags, paths, queries, gallery):
    rk = {}
    for tag in tags:
        rk[tag] = ranked_from_sim(paths[tag], queries, gallery)
    return rk


def truncate(rk_dict, tag, K):
    """把某成员的完整排名截断为 top-K（RRF 的 member_topk）。"""
    return {q: lst[:K] for q, lst in rk_dict[tag].items()}


def fuse(rankings, weights, k, topk, member_topk):
    """按 member_topk 截断各成员排名后做加权 RRF（weights 与 rankings 的键顺序一致）。"""
    lists = [truncate(rankings, t, member_topk) for t in rankings]
    return rrf(lists, list(weights), k=k, topk=topk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=str(ROOT.parent / "question6-data" / "traindata" / "local_val_task.json"))
    ap.add_argument("--gt", default=str(ROOT.parent / "question6-data" / "traindata" / "local_val_gt.json"))
    ap.add_argument("--base_dir", default=str(ROOT / "sims" / "localval"))
    ap.add_argument("--base_preset", default="final", choices=sorted(PRESETS))
    ap.add_argument("--cand_dir", default="", help="候选 sim_<tag>.pt 所在目录")
    ap.add_argument("--cand", nargs="*", default=[],
                    help="方案改动，写 tag=权重（tag 可为基线成员=改权重/设 0 剔除，或候选目录里的新成员；"
                         "缺省=目录下全部 sim_*.pt 以 --cand_weight 加入）")
    ap.add_argument("--cand_weight", type=float, default=1.0)
    ap.add_argument("--k_prod", type=int, default=3)
    ap.add_argument("--k_scan", type=int, nargs="+", default=[1, 3, 5, 10, 20, 40])
    ap.add_argument("--gate", type=float, default=0.003, help="G1 增益门槛")
    ap.add_argument("--tol", type=float, default=1e-4, help="G2 浮点容差")
    ap.add_argument("--member_topk", type=int, default=100, help="各成员输入融合的排名深度")
    ap.add_argument("--topk", type=int, default=10, help="每条 query 保留候选数")
    ap.add_argument("--joint", action="store_true", help="把所有候选作为一个联合方案评估（而非逐个）")
    args = ap.parse_args()

    task = load_json(args.task)
    queries, gallery = task["queries"], task["gallery"]
    gt = {k: set(v) for k, v in load_json(args.gt).items()}
    folds = fold_split(queries)
    print(f"[任务] queries={len(queries)} gallery={len(gallery)} "
          f"fold0={len(folds[0])} fold1={len(folds[1])}")

    # ---- 基线成员 ----
    base_dir = Path(args.base_dir)
    preset = PRESETS[args.base_preset]
    base_tags = [t for t in preset if t in BASE_MEMBERS]
    base_paths = {t: base_dir / BASE_MEMBERS[t] for t in base_tags}
    for t, p in base_paths.items():
        if not p.exists():
            raise SystemExit(f"[ERROR] 缺基线产物 {p}")
    base_rk = build_rankings(base_tags, base_paths, queries, gallery)
    base_w = [preset[t] for t in base_tags]
    print(f"[基线] preset={args.base_preset} 成员 {len(base_tags)} 个: "
          + ", ".join(f"{t}({preset[t]})" for t in base_tags))
    print("[基线] 单模 Final（top-10，仅供对照）")
    for t in base_tags:
        r = evaluate({q: base_rk[t][q][:10] for q in base_rk[t]}, gt, queries)
        print(f"  {t:<24} {r['overall']:.4f}   "
              + "  ".join(f"{d} {r[d]['score']:.4f}" for d in DIRECTIONS))

    # ---- 基线锚点自校验（k=3）----
    base_by_k = {}
    for k in args.k_scan:
        bf = fuse(base_rk, base_w, k=float(k), topk=args.topk, member_topk=args.member_topk)
        be = evaluate(bf, gt, queries)
        bf0, bf1 = fold_eval(bf, gt, folds)
        base_by_k[k] = {"eval": be, "fold0": bf0, "fold1": bf1}
    pick = base_by_k[args.k_prod] if args.k_prod in base_by_k else base_by_k[args.k_scan[0]]
    base_eval = pick["eval"]
    b0, b1 = pick["fold0"], pick["fold1"]
    got = {"overall": base_eval["overall"], "fold0": b0, "fold1": b1}
    print(f"\n[锚点自校验] k={args.k_prod} member_topk={args.member_topk}")
    ok_anchor = True
    for key in ("overall", "fold0", "fold1"):
        exp, act = ANCHORS[key], got[key]
        good = abs(act - exp) <= ANCHOR_TOL
        ok_anchor &= good
        print(f"  {key:<8} 期望 {exp:.4f}  实测 {act:.4f}  "
              f"差 {act - exp:+.4f}  {'OK' if good else 'FAIL'}")
    print(f"  基线方向: " + "  ".join(f"{d} {base_eval[d]['score']:.4f}" for d in DIRECTIONS))
    if not ok_anchor:
        print("  ！锚点不符：口径/产物已变，请勿采信本次筛选结论")
    print("[基线] k 扫描（参考值；增量门槛一律相对同 k 的基线）")
    for k in args.k_scan:
        b = base_by_k[k]
        print(f"  k={k:<3} Final {b['eval']['overall']:.4f}  "
              f"fold0 {b['fold0']:.4f}  fold1 {b['fold1']:.4f}")

    if not args.cand_dir:
        print("\n[提示] 未指定 --cand_dir，仅完成基线自校验。")
        return

    # ---- 候选/变体方案 ----
    # cand_spec 的键可以是基线成员（改权重，0 = 剔除，用于「替换」）或候选目录里的新成员（新增）。
    cand_dir = Path(args.cand_dir)
    cand_spec = {}
    for item in args.cand:
        name, _, w = item.partition("=")
        cand_spec[name] = float(w) if w else args.cand_weight
    if not cand_spec:
        cand_spec = {p.name[len("sim_"):-len(".pt")]: args.cand_weight
                     for p in sorted(cand_dir.glob("sim_*.pt"))}
    if not cand_spec:
        raise SystemExit(f"[ERROR] {cand_dir} 下没有候选 sim_*.pt")

    new_tags = [t for t in cand_spec if t not in BASE_MEMBERS]
    bad = [t for t in cand_spec if t in BASE_MEMBERS and t not in preset]
    if bad:
        raise SystemExit(f"[ERROR] 改权重的成员不在预设 {args.base_preset}: {bad}")

    # 变体方案的成员排名：基线成员复用 base_rk，新成员从 cand_dir 加载
    if new_tags:
        new_paths = {}
        for t in new_tags:
            p = cand_dir / f"sim_{t}.pt"
            if not p.exists():
                raise SystemExit(f"[ERROR] 缺候选产物 {p}")
            new_paths[t] = p
        new_rk = build_rankings(new_tags, new_paths, queries, gallery)
    else:
        new_rk = {}

    if new_tags:
        print(f"\n[候选] 新成员 {len(new_tags)} 个: " + ", ".join(new_tags))
        ref_tag = "convnext_t_ep080" if "convnext_t_ep080" in base_rk else None
        ref_eval = (evaluate({q: base_rk[ref_tag][q][:10] for q in base_rk[ref_tag]},
                             gt, queries)["overall"] if ref_tag else None)
        for t in new_tags:
            r = evaluate({q: new_rk[t][q][:10] for q in new_rk[t]}, gt, queries)
            delta = (f"  (vs 同骨干基准 {ref_tag} {r['overall'] - ref_eval:+.4f})"
                     if ref_eval is not None else "")
            print(f"  {t:<24} 单模 Final {r['overall']:.4f}   "
                  + "  ".join(f"{d} {r[d]['score']:.4f}" for d in DIRECTIONS) + delta)

    def scheme_rankings(spec):
        """变体方案 = 基线预设叠加 spec（新成员加入 / 已有成员改权重，权重 0 即剔除）。"""
        rk, w = {}, {}
        for t, wt in preset.items():
            wt = spec.get(t, wt)
            if wt <= 0:
                continue
            rk[t] = base_rk[t]
            w[t] = wt
        for t, wt in spec.items():
            if t in BASE_MEMBERS or wt <= 0:
                continue
            rk[t] = new_rk[t]
            w[t] = wt
        return rk, w

    # ---- k 扫描 ----
    def scan(extra_w):
        rk, w = scheme_rankings(extra_w)
        rows = []
        for k in args.k_scan:
            ref = base_by_k[k]
            fused = fuse(rk, [w[t] for t in rk], k=float(k),
                         topk=args.topk, member_topk=args.member_topk)
            e = evaluate(fused, gt, queries)
            f0, f1 = fold_eval(fused, gt, folds)
            rows.append({
                "k": k, "overall": e["overall"], "fold0": f0, "fold1": f1,
                "d_overall": e["overall"] - ref["eval"]["overall"],
                "d_fold0": f0 - ref["fold0"], "d_fold1": f1 - ref["fold1"],
                "dirs": {d: e[d]["score"] for d in DIRECTIONS},
                "dirs_base": {d: ref["eval"][d]["score"] for d in DIRECTIONS},
                "n_members": len(rk),
            })
        return rows

    def report(name, rows, spec):
        print(f"\n===== 变体方案：{name} =====")
        print(f"  改动: " + ", ".join(f"{t}={w}" for t, w in sorted(spec.items()))
              + f"   (成员数 {rows[0]['n_members']})")
        print(f"  {'k':>3}{'Final':>9}{'ΔFinal':>9}{'fold0':>9}{'Δfold0':>9}"
              f"{'fold1':>9}{'Δfold1':>9}   G1  G2")
        for r in rows:
            g1 = "OK " if (r["d_fold0"] >= args.gate and r["d_fold1"] >= args.gate) else "   "
            g2 = "OK " if (r["d_fold0"] >= -args.tol and r["d_fold1"] >= -args.tol) else "NEG"
            print(f"  {r['k']:>3}{r['overall']:>9.4f}{r['d_overall']:>+9.4f}{r['fold0']:>9.4f}"
                  f"{r['d_fold0']:>+9.4f}{r['fold1']:>9.4f}{r['d_fold1']:>+9.4f}   {g1} {g2}")
        prod = next(r for r in rows if r["k"] == args.k_prod)
        g1_ok = prod["d_fold0"] >= args.gate and prod["d_fold1"] >= args.gate
        worst = min(min(r["d_fold0"], r["d_fold1"]) for r in rows)
        g2_ok = worst >= -args.tol
        print(f"  G1(k={args.k_prod} 两折 >= +{args.gate}): "
              f"{'通过' if g1_ok else '未通过'}  (fold0 {prod['d_fold0']:+.4f} / fold1 {prod['d_fold1']:+.4f})")
        print(f"  G2(k 扫描任何折不为负，容差 {args.tol}): "
              f"{'通过' if g2_ok else '未通过'}  (最差折 {worst:+.4f})")
        print(f"  k={args.k_prod} 方向拆解: "
              + "  ".join(f"{d} {prod['dirs_base'][d]:.4f}->{prod['dirs'][d]:.4f}"
                          f"({prod['dirs'][d] - prod['dirs_base'][d]:+.4f})" for d in DIRECTIONS))
        wk = min(rows, key=lambda r: min(r["d_fold0"], r["d_fold1"]))
        print(f"  最差点 k={wk['k']}（最差折 {worst:+.4f}，"
              f"fold0 {wk['d_fold0']:+.4f} / fold1 {wk['d_fold1']:+.4f}）方向拆解: "
              + "  ".join(f"{d} {wk['dirs_base'][d]:.4f}->{wk['dirs'][d]:.4f}"
                          f"({wk['dirs'][d] - wk['dirs_base'][d]:+.4f})" for d in DIRECTIONS))
        return g1_ok, g2_ok, worst

    results = {}
    if args.joint:
        name = " + ".join(f"{t}={cand_spec[t]}" for t in cand_spec)
        results[name] = report(name, scan(dict(cand_spec)), dict(cand_spec))
    else:
        for t in cand_spec:
            results[t] = report(t, scan({t: cand_spec[t]}), {t: cand_spec[t]})

    # ---- 门槛汇总 ----
    print("\n===== 三重门槛汇总 =====")
    g1_pass = [n for n, (a, _, _) in results.items() if a]
    g2_fail = [n for n, (_, b, _) in results.items() if not b]
    both = [n for n, (a, b, _) in results.items() if a and b]
    print(f"  G1 通过: {g1_pass or '无'}")
    print(f"  G2 通过: {[n for n in results if n not in g2_fail] or '无'}")
    print(f"  G1+G2 同时通过（进入 G3 人工复核）: {both or '无'}")
    print("\n[被门槛2拦下] " + ("无" if not g2_fail else
          "; ".join(f"{n}(最差折 {results[n][2]:+.4f})" for n in g2_fail)))


if __name__ == "__main__":
    main()