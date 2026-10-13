# -*- coding: utf-8 -*-
"""S5 转导互反重排：忠实重建（对齐 %TEMP%\\s5_final.py 原件）+ ms 变体生成。

配方（wave13 S5）：
    基座：11 个现役 pred 文件 RRF（k=10、top-100 榜、O2S 覆盖 mos75=1.5/shipvit=0.5/cnx_t=0.25），
          保真自检 = prediction_ctrl_pre_s5.json（sha 9C7A42B9）。
    R2 项：10 个现役权重稠密 sim（7 ship_reid full + 3 SDF-Net full）**先求均值 smean**，
          再对每个 query 在**目标模态内 min-max 归一化**（lo/hi 取自目标模态列），
          score(c) = RRF_base(c) + 0.5 * norm(c)，仅作用于 RRF 池候选（原脚本语义）。
    base75 无稠密 sim，仅经基座贡献；合法掩码 = 目标模态且非自身图像。

保真门：重跑须复现 prediction_s5.json（sha 237F1E77）267/267。

用法（cwd = 工程根）：
    ship_reid_vit\\venv\\Scripts\\python.exe scripts\\s5_rebuild.py --fidelity
    ... --emit                       # 重建 S5（对照）
    ... --emit-ms                    # S5'：shipvit 换 multiscale 版（基座 pred + 稠密 sim 同步换）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT.parent  # H:\Ship-Re-Identification
TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}
LAMBDA = 0.5  # 原脚本硬编码
K = 10.0

# 10 个稠密 sim（现役权重；base75 无稠密 sim）
DENSE = {
    "sdfnet_mos75_full": "sims/test_rerank/sdfsim_mos75_full/sim.pt",
    "sdfnet_ep45_full": "sims/test_rerank/sdfsim_ep45_full/sim.pt",
    "sdfnet_ep25_full": "sims/test_rerank/sdfsim_ep25_full/sim.pt",
    "ship_reid_vit_full": "sims/test_rerank_full/sim_shipvit_full.pt",
    "ship_reid_convnext_t_full": "sims/test_rerank_full/sim_convnext_t_full.pt",
    "ship_reid_convnext_t_strong_cm_full": "sims/test_rerank_full/sim_convnext_t_strong_cm_full.pt",
    "ship_reid_convnext_s_pure_full": "sims/test_rerank_full/sim_convnext_s_pure_full.pt",
    "ship_reid_swin_s_v2_full": "sims/test_rerank_full/sim_swin_s_v2_full.pt",
    "ship_reid_dinov2_b_ep040": "sims/test_rerank_full/sim_dinov2_b_ep040.pt",
    "ship_reid_dinov2_b_ep060": "sims/test_rerank_full/sim_dinov2_b_ep060.pt",
}


def load_cfg() -> dict:
    with open(ROOT / "configs" / "reproduce.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def load_task() -> dict:
    with open(resolve(load_cfg()["paths"]["test_task"]), encoding="utf-8") as f:
        return json.load(f)


def legality(task: dict):
    """返回 (q_ids, g_ids, legal 候选索引 list, q_types)。合法 = 目标模态 且 非自身图像。"""
    queries, gallery = task["queries"], task["gallery"]
    q_ids = [q["query_id"] for q in queries]
    g_ids = [g["image_id"] for g in gallery]
    g_mods = [g["modality"] for g in gallery]
    g_paths = [g["image_path"] for g in gallery]
    masks = []
    for q in queries:
        want = TARGET_MOD[q["query_type"]]
        masks.append(torch.tensor(
            [i for i in range(len(gallery)) if g_mods[i] == want and g_paths[i] != q["image_path"]],
            dtype=torch.long))
    return q_ids, g_ids, masks, [q["query_type"] for q in queries]


def member_weights(cfg: dict, preset: str):
    """返回 (成员名列表, 基础权重 dict, O2S 覆盖后权重 dict)。"""
    spec = dict(cfg["ensemble"]["presets"][preset])
    per_dir = spec.pop("per_direction", None) or {}
    names = list(spec)
    w = {n: float(spec[n]) for n in names}
    wo = dict(w)
    for n, v in (per_dir.get("O2S") or {}).items():
        if n in wo:
            wo[n] = float(v)
    return names, w, wo


def pred_path(cfg: dict, name: str) -> str:
    return [m for m in cfg["ensemble"]["members"] if m["name"] == name][0]["pred"]


def rrf_base(cfg: dict, preset: str, q_ids, q_types):
    """RRF 基座分数 {qid: {gid: score}}（k=10；O2S 用覆盖权重）。"""
    names, w, wo = member_weights(cfg, preset)
    lists = {}
    for n in names:
        with open(resolve(pred_path(cfg, n)), encoding="utf-8") as f:
            lists[n] = json.load(f)
    scores = {}
    for qid, d in zip(q_ids, q_types):
        wt = wo if d == "O2S" else w
        acc = {}
        for n in names:
            for rank, gid in enumerate(lists[n].get(qid, [])[:100], 1):
                acc[gid] = acc.get(gid, 0.0) + wt[n] / (K + rank)
        scores[qid] = acc
    return scores


def mean_sim(sim_rel_paths) -> np.ndarray:
    """10 成员稠密 sim 的逐元素均值（float64）。"""
    acc = None
    for rel in sim_rel_paths:
        sim = torch.load(resolve(rel), map_location="cpu").numpy().astype(np.float64)
        acc = sim if acc is None else acc + sim
    return acc / len(sim_rel_paths)


def s5_pred(scores, q_ids, g_ids, masks, smean: np.ndarray, topk=10):
    """S5：score = RRF + 0.5 * minmax_within_target(smean)，仅对 RRF 池候选加 R2。"""
    gid2gi = {g: i for i, g in enumerate(g_ids)}
    out = {}
    for qi, qid in enumerate(q_ids):
        m = masks[qi].numpy()
        vals = smean[qi][m]
        lo, hi = vals.min(), vals.max()
        norm = (smean[qi] - lo) / (hi - lo + 1e-12)
        acc = dict(scores[qid])
        for gid in acc:
            acc[gid] += LAMBDA * norm[gid2gi[gid]]
        out[qid] = [g for g, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def compare(pred: dict, ref_path: Path):
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    same = sum(1 for q in pred if pred[q] == ref.get(q))
    t1 = sum(1 for q in pred if pred[q][:1] == ref.get(q, [])[:1])
    inter = np.mean([len(set(pred[q]) & set(ref.get(q, []))) for q in pred])
    return same, t1, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="final_full_b75_cm_dino_nocnxs_dir")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--emit", action="store_true", help="重建 S5（对照输出）")
    ap.add_argument("--emit-ms", action="store_true", help="S5'：shipvit 换 multiscale 版")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_cfg()
    task = load_task()
    q_ids, g_ids, masks, q_types = legality(task)

    ctrl = WS / "prediction_ctrl_pre_s5.json"
    s5_ref = WS / "prediction_s5.json"

    # ---- 保真门 1：RRF 基座 ----
    if args.emit_ms:
        cfg = json.loads(json.dumps(cfg))
        for m in cfg["ensemble"]["members"]:
            if m["name"] == "ship_reid_vit_full":
                m["pred"] = "sims/test_full/ms_top100_shipvit_full.json"
    scores = rrf_base(cfg, args.preset, q_ids, q_types)
    if ctrl.exists():
        base_pred = {qid: [g for g, _ in sorted(scores[qid].items(), key=lambda kv: -kv[1])[:10]]
                     for qid in q_ids}
        same, t1, _ = compare(base_pred, ctrl)
        print(f"[保真门1] RRF 基座 vs prediction_ctrl_pre_s5.json: 全同 {same}/267, top-1 同 {t1}/267")

    # ---- R2 项 ----
    dense = dict(DENSE)
    if args.emit_ms:
        dense["ship_reid_vit_full"] = "sims/test_rerank_full_ms/sim_shipvit_full_ms.pt"
    smean = mean_sim(list(dense.values()))

    # ---- 保真门 2 / 输出 ----
    pred = s5_pred(scores, q_ids, g_ids, masks, smean)
    if s5_ref.exists() and not args.emit_ms:
        same, t1, inter = compare(pred, s5_ref)
        print(f"[保真门2] 重建 S5 vs prediction_s5.json: 全同 {same}/267, top-1 同 {t1}/267, 平均交集 {inter:.3f}/10")
    if args.emit_ms:
        same, t1, inter = compare(pred, ctrl)
        print(f"[S5'] vs 现役基座: top-1 同 {t1}/267, 平均交集 {inter:.3f}/10")
        flips = [q for q in q_ids if pred[q][0] != json.loads(ctrl.read_text(encoding='utf-8'))[q][0]]
        print(f"[S5'] top-1 翻转 {len(flips)} 条")

    if args.emit or args.emit_ms:
        out = Path(args.out) if args.out else (
            WS / ("prediction_s5_ms.json" if args.emit_ms else "prediction_s5_rebuild.json"))
        out.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written: {out}")


if __name__ == "__main__":
    main()