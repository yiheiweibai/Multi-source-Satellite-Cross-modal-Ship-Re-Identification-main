# -*- coding: utf-8 -*-
"""Phase 1 集成层消融：成员加多尺度特征（ms）对 RRF 集成的净效应。

对比（同一 RRF 配方，唯一变量 = ship_reid 成员的特征提取协议）：
  基线臂 base：5 个缩水 ship_reid 成员 = tta+rerank（现役协议）
              -> sims/localval_ab/top100_<tag>_base.json
  多尺度臂 ms：同成员 = tta+rerank+multiscale(0.7/0.2/0.1)
              -> sims/localval_ab/top100_<tag>_ms.json
SDF-Net 3 个成员两臂共用（由 sims/localval/sdfsim_*/sim.pt 现场转 top-100 排名表）。

融合：RRF k=10 / member_topk=100 / 权重 = final 预设 [1,1,1,1,1,0.5,0.75,1]（扁平，无方向覆盖）。
评测：local_val 1450 query；方向加权 0.45/0.45/0.10；parity 两折（各方向内部 folds[i%2]）。

用法（工程根目录，ship_reid_vit 的 venv）：
    cd ship_reid_vit && .\\venv\\Scripts\\python.exe ..\\scripts\\phase1_ensemble_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent  # Multi-source-...-main 工程根
sys.path.insert(0, str(ROOT / "ship_reid_vit"))

from data import load_task  # noqa: E402
from data.dataset import MODALITY_ID  # noqa: E402
from inference import indices_from_sim  # noqa: E402
from utils.metrics import evaluate_submission  # noqa: E402

TASK = ROOT.parent / "question6-data" / "traindata" / "local_val_task.json"
GT = ROOT.parent / "question6-data" / "traindata" / "local_val_gt.json"
AB_DIR = ROOT / "sims" / "localval_ab"

MEMBER_TOPK = 100
RRF_K = 10.0
WEIGHTS = {
    "sdfnet_mos75": 1.0,
    "sdfnet_ep45": 1.0,
    "sdfnet_ep25": 1.0,
    "shipvit": 1.0,
    "convnext_t": 1.0,
    "convnext_s": 0.5,
    "convnext_s_pure": 0.75,
    "swin_s_v2": 1.0,
}
ORDER = list(WEIGHTS)
SHIP_TAGS = ("shipvit", "convnext_t", "convnext_s", "convnext_s_pure", "swin_s_v2")
SDF_SIMS = {
    "sdfnet_mos75": "sims/localval/sdfsim_mos75/sim.pt",
    "sdfnet_ep45": "sims/localval/sdfsim_ep45/sim.pt",
    "sdfnet_ep25": "sims/localval/sdfsim_ep25/sim.pt",
}


def rrf(members: list, weights: list, k: float, topk: int = 10) -> dict:
    """与 scripts/rrf_fuse.py::rrf 同式：score(c)=Σ w/(k+rank)。"""
    out = {}
    for qid in members[0]:
        s = {}
        for lst, w in zip(members, weights):
            for rank, iid in enumerate(lst.get(qid, []), 1):
                s[iid] = s.get(iid, 0.0) + w / (k + rank)
        out[qid] = [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def parity_folds(queries: list) -> dict:
    folds = {0: [], 1: []}
    for d in ("O2S", "S2O", "O2O"):
        sub = [q for q in queries if q["query_type"] == d]
        for i, q in enumerate(sub):
            folds[i % 2].append(q)
    return folds


def main() -> None:
    task = load_task(str(TASK))
    queries, gallery = task["queries"], task["gallery"]
    with open(GT, encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt = {k: set(v) if isinstance(v, list) else {v} for k, v in gt_raw.items()}
    g_mods = torch.tensor([MODALITY_ID[g["modality"]] for g in gallery], dtype=torch.long)
    print(f"local_val: {len(queries)} query / {len(gallery)} gallery")

    # SDF-Net 成员：dense sim -> top-100 排名表（两臂共用）
    sdf_preds = {}
    for name, rel in SDF_SIMS.items():
        sim = torch.load(ROOT / rel, map_location="cpu").float()
        assert tuple(sim.shape) == (len(queries), len(gallery)), \
            f"{name}: {tuple(sim.shape)}"
        idx = indices_from_sim(sim, g_mods, queries, gallery, topk=MEMBER_TOPK)
        sdf_preds[name] = {
            queries[qi]["query_id"]: [gallery[gi]["image_id"] for gi in idx[qi].tolist()]
            for qi in range(len(queries))
        }
        print(f"[sdf] {name:<14} top-{MEMBER_TOPK} 排名表构建完成")

    # ship_reid 两臂成员表
    arms = {}
    for arm, suffix in (("base", "_base.json"), ("ms", "_ms.json")):
        preds = dict(sdf_preds)
        for tag in SHIP_TAGS:
            p = AB_DIR / f"top100_{tag}{suffix}"
            if not p.exists():
                print(f"[中止] 缺少成员文件: {p}")
                return
            with open(p, encoding="utf-8") as f:
                preds[tag] = json.load(f)
        arms[arm] = preds

    # 成员级诊断：top-1 变化数（base -> ms）
    print("\n== 成员 top-1 变化（base -> ms）==")
    for tag in SHIP_TAGS:
        ch = sum(1 for q in queries
                 if arms["base"][tag][q["query_id"]][0] != arms["ms"][tag][q["query_id"]][0])
        print(f"  {tag:<16} {ch}/{len(queries)}")

    # 融合 + 评测
    folds = parity_folds(queries)
    results = {}
    for arm in ("base", "ms"):
        members = [arms[arm][n] for n in ORDER]
        w = [WEIGHTS[n] for n in ORDER]
        fused = rrf(members, w, RRF_K, topk=10)
        full = evaluate_submission(fused, gt, queries, topk=10)
        f0 = evaluate_submission(fused, gt, folds[0], topk=10)["overall"]
        f1 = evaluate_submission(fused, gt, folds[1], topk=10)["overall"]
        results[arm] = {"full": full, "fold0": f0, "fold1": f1, "fused": fused}

    print("\n" + "=" * 66)
    print(f"{'':10}{'基线 base':>16}{'多尺度 ms':>16}{'Δ(ms-base)':>16}")
    for d in ("O2S", "S2O", "O2O"):
        b = results["base"]["full"][d]["score"]
        m = results["ms"]["full"][d]["score"]
        print(f"{d:<10}{b:>16.4f}{m:>16.4f}{m - b:>+16.4f}")
    print("-" * 66)
    b = results["base"]["full"]["overall"]
    m = results["ms"]["full"]["overall"]
    print(f"{'Final':<10}{b:>16.4f}{m:>16.4f}{m - b:>+16.4f}")
    print(f"{'fold0':<10}{results['base']['fold0']:>16.4f}{results['ms']['fold0']:>16.4f}"
          f"{results['ms']['fold0'] - results['base']['fold0']:>+16.4f}")
    print(f"{'fold1':<10}{results['base']['fold1']:>16.4f}{results['ms']['fold1']:>16.4f}"
          f"{results['ms']['fold1'] - results['base']['fold1']:>+16.4f}")
    print("=" * 66)
    print("门槛：Δ(Final) >= +0.003 且 fold0/fold1 双折同向为正")

    for arm in ("base", "ms"):
        out = ROOT / f"prediction_localval_ab_{arm}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results[arm]["fused"], f, ensure_ascii=False, indent=1)
        print(f"已保存 {out}")

    # ---- 诊断 A：成员同质化程度（两臂成员间 top-1 平均一致率）----
    print("\n== 同质化诊断：5 成员两两 top-1 平均一致率 ==")
    for arm in ("base", "ms"):
        agrees = []
        for i in range(len(SHIP_TAGS)):
            for j in range(i + 1, len(SHIP_TAGS)):
                a, b = SHIP_TAGS[i], SHIP_TAGS[j]
                c = sum(1 for q in queries
                        if arms[arm][a][q["query_id"]][0] == arms[arm][b][q["query_id"]][0])
                agrees.append(c / len(queries))
        print(f"  {arm:<5} {sum(agrees) / len(agrees):.4f}")

    # ---- 诊断 B：32 种子集扫描（次 = 用 ms 版成员）----
    # 门槛 G1：parity / 前后半 / 相位4 三种折分割，每种的 fold0 与 fold1 增益均 >= +0.003
    def split_variants(qs):
        out = {}
        for name, fn in (("parity", lambda i, n: i % 2),
                         ("halves", lambda i, n: 0 if i < (n + 1) // 2 else 1),
                         ("phase4", lambda i, n: 0 if i % 4 < 2 else 1)):
            folds = {0: [], 1: []}
            for d in ("O2S", "S2O", "O2O"):
                sub = [q for q in qs if q["query_type"] == d]
                for i, q in enumerate(sub):
                    folds[fn(i, len(sub))].append(q)
            out[name] = folds
        return out

    def arm_eval(preds, k=RRF_K):
        members = [preds[n] for n in ORDER]
        w = [WEIGHTS[n] for n in ORDER]
        fused = rrf(members, w, k, topk=10)
        return fused

    print("\n== 子集扫描（32 组合，基准=base 臂同配方）==")
    print(f"{'组合':<44}{'Final':>9}{'Δ':>9}{'分折最小增益(3折分割)':>22}")
    split_cache = split_variants(queries)
    base_fused = results["base"]["fused"]
    base_full = results["base"]["full"]["overall"]
    candidates = []
    for mask in range(32):
        use_ms = [tag for i, tag in enumerate(SHIP_TAGS) if (mask >> i) & 1]
        preds = {n: (arms["ms"][n] if n in use_ms else arms["base"][n]) for n in ORDER}
        fused = arm_eval(preds)
        full = evaluate_submission(fused, gt, queries, topk=10)["overall"]
        deltas = []
        for sname, folds in split_cache.items():
            b0 = evaluate_submission(base_fused, gt, folds[0], topk=10)["overall"]
            m0 = evaluate_submission(fused, gt, folds[0], topk=10)["overall"]
            b1 = evaluate_submission(base_fused, gt, folds[1], topk=10)["overall"]
            m1 = evaluate_submission(fused, gt, folds[1], topk=10)["overall"]
            deltas.extend([m0 - b0, m1 - b1])
        min_d = min(deltas)
        label = "全base" if not use_ms else ("全ms" if len(use_ms) == 5 else "+".join(use_ms))
        flag = " *G1*" if min_d >= 0.003 else ""
        print(f"{label:<44}{full:>9.4f}{full - base_full:>+9.4f}{min_d:>+22.4f}{flag}")
        if use_ms:
            candidates.append((label, full, full - base_full, min_d, use_ms))

    passed = [c for c in candidates if c[3] >= 0.003]
    print(f"\n通过 G1（三折分割全部双折 >= +0.003）的子集: "
          f"{len(passed)} 个" + (f" -> {[c[0] for c in passed]}" if passed else "（无）"))
    if passed:
        print("注：32 组合存在多重检验偏差，通过者仍须 k 扫描复核：")
        for label, full, d, min_d, use_ms in passed:
            preds = {n: (arms["ms"][n] if n in use_ms else arms["base"][n]) for n in ORDER}
            folds0, folds1 = split_cache["parity"][0], split_cache["parity"][1]
            ks = []
            for k in (1, 3, 5, 7, 10, 15, 20, 40):
                f = arm_eval(preds, k=float(k))
                f0 = evaluate_submission(f, gt, folds0, topk=10)["overall"]
                b0 = evaluate_submission(base_fused, gt, folds0, topk=10)["overall"]
                f1 = evaluate_submission(f, gt, folds1, topk=10)["overall"]
                b1 = evaluate_submission(base_fused, gt, folds1, topk=10)["overall"]
                ks.append((k, min(f0 - b0, f1 - b1)))
            print(f"  {label}: " + " ".join(f"k={k}:{v:+.4f}" for k, v in ks))


if __name__ == "__main__":
    main()