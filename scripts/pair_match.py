# -*- coding: utf-8 -*-
"""闭合配对 predictor 求解器（入库版；对应 507F895D / 官方 0.849 的提交管线）。

锚构造见 pair_verify.py（md5 字节复制 + 7 模型均值同模态近邻兜底），本脚本消费其锚资产。

结构（已实测）：
  · 测试集 O2S 120 条与 S2O 120 条 query 共享同一身份池（现役 top-1 落点 O2S 101/120∈A_s、
    S2O 99/120∈A_o、O2O 27/27∈A_o；随机仅 13.3%）。
  · A_o[i] = O2S i 的光学 gallery 锚（93 精确字节复制 + 27 同模态近邻，互异）；
    A_s[j] = S2O j 的 SAR gallery 锚（120 全精确，互异）。
  · 配对 (i,j) 正确 ⟺ O2S i 的答案含 A_s[j]、S2O j 的答案含 A_o[i]。

证据通道（全部来自现役产物，零新推理）：
  sim ×10 模型 ×2 向：sim_m[O2S i, A_s[j]] / sim_m[S2O j, A_o[i]]（267×1800 稠密 sim）
  fq  × 7 模型 ×2 向：cos(qf_m[q_i], gf_m[A_s[j]]) / cos(qf_m[q_j], gf_m[A_o[i]])
  fg  × 7 模型 ×1 向：cos(gf_m[A_o[i]], gf_m[A_s[j]])（gallery 侧锚对）

融合：各通道 120×120 全局秩 RRF 1/(10+rank) → 加权求和 → scipy 匈牙利最大权匹配。

锚资产：优先取本脚本同目录 pair_anchors.json（已随仓库入库），回退 %TEMP%。

用法：
  python pair_match.py          # 分析 + 自检（基线默认 WS/prediction.json）
  python pair_match.py --base prediction_s5.json --diag --emit --out pair_reemit.json
      # 0.849 版对照基线 = prediction_s5.json（= 替换前现役）；--diag 落盘逐行置信度表
      # （outputs/pair_diag.csv）；--out 用于重放对比（重放结果应与 prediction_closedpair.json
      # 267/267 一致）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

WS = Path(r"h:\Ship-Re-Identification")
TT = WS / "question6-data" / "preliminary-round-test-data"
PROJ = WS / "Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main"
TEMP = Path(os.environ["TEMP"])
K_RRF = 10.0

DENSE = {
    "sdfnet_mos75_full": "sims/test_rerank/sdfsim_mos75_full/sim.pt",
    "sdfnet_ep45_full": "sims/test_rerank/sdfsim_ep45_full/sim.pt",
    "sdfnet_ep25_full": "sims/test_rerank/sdfsim_ep25_full/sim.pt",
    "shipvit_full": "sims/test_rerank_full/sim_shipvit_full.pt",
    "convnext_t_full": "sims/test_rerank_full/sim_convnext_t_full.pt",
    "convnext_t_strong_cm_full": "sims/test_rerank_full/sim_convnext_t_strong_cm_full.pt",
    "convnext_s_pure_full": "sims/test_rerank_full/sim_convnext_s_pure_full.pt",
    "swin_s_v2_full": "sims/test_rerank_full/sim_swin_s_v2_full.pt",
    "dinov2_b_ep040": "sims/test_rerank_full/sim_dinov2_b_ep040.pt",
    "dinov2_b_ep060": "sims/test_rerank_full/sim_dinov2_b_ep060.pt",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="另存 prediction_closedpair.json")
    ap.add_argument("--base", default="", help="对照基线 prediction（默认 WS/prediction.json）")
    ap.add_argument("--out", default="", help="emit 输出路径（默认 WS/prediction_closedpair.json）")
    ap.add_argument("--diag", action="store_true", help="落盘逐行置信度表 outputs/pair_diag.csv")
    args = ap.parse_args()

    # ---------- 锚与 task ----------
    apath = next((p for p in (Path(__file__).resolve().parent / "pair_anchors.json",
                              TEMP / "pair_anchors.json") if p.exists()), None)
    assert apath is not None, "pair_anchors.json 未找到（脚本同目录与 TEMP 均无）"
    print("[A0] 锚资产: %s" % apath)
    anchors = json.loads(apath.read_text(encoding="utf-8"))
    A_o, A_s, copy_src = anchors["A_o"], anchors["A_s"], anchors["copy_src"]
    o2s_ids, s2o_ids = sorted(A_o), sorted(A_s)
    n = len(o2s_ids)
    assert n == len(s2o_ids) == 120, (n, len(s2o_ids))

    task = json.loads((TT / "task.json").read_text(encoding="utf-8"))
    gmod = {g["image_id"]: g["modality"] for g in task["gallery"]}
    o2o_ids = [q["query_id"] for q in task["queries"] if q["query_type"] == "O2O"]
    print("[A] 锚: A_o %d（精确 %d）/ A_s %d（精确 %d）| O2O %d 条（不动）" % (
        len(A_o), sum(1 for q in o2s_ids if q in copy_src),
        len(A_s), sum(1 for q in s2o_ids if q in copy_src), len(o2o_ids)))
    print("    A_s 全 SAR: %s | A_o 全 optical: %s" % (
        all(gmod[A_s[j]] == "sar" for j in s2o_ids),
        all(gmod[A_o[i]] == "optical" for i in o2s_ids)))

    # ---------- feat / sim ----------
    fd = PROJ / "sims" / "test_rerank_full"
    feats = {p.stem[len("feat_"):]: torch.load(p, map_location="cpu")
             for p in sorted(fd.glob("feat_*.pt"))}
    ref = feats[sorted(feats)[0]]
    qids = [str(x) for x in ref["q_ids"]]
    gids = [str(x) for x in ref["g_ids"]]
    for nm, m in feats.items():
        assert [str(x) for x in m["q_ids"]] == qids and [str(x) for x in m["g_ids"]] == gids, nm
    qi = {x: i for i, x in enumerate(qids)}
    gi = {x: i for i, x in enumerate(gids)}

    sims = {}
    for nm, rel in DENSE.items():
        S = torch.load(PROJ / rel, map_location="cpu").float().numpy()
        assert S.shape == (len(qids), len(gids)), (nm, S.shape)
        sims[nm] = S

    row_o2s = np.array([qi[q] for q in o2s_ids])
    row_s2o = np.array([qi[q] for q in s2o_ids])
    col_sar = np.array([gi[A_s[j]] for j in s2o_ids])
    col_opt = np.array([gi[A_o[i]] for i in o2s_ids])

    # ---------- 对齐抽查：精确复制件的同模态 sim 应为 1（未掩码）或 0（sdf 系掩码，属预期） ----------
    print("[B] 对齐抽查（复制件同模态格；sdf 系整体掩码为 0，已实测确认）:")
    ok = True
    for nm, S in sims.items():
        a = np.mean([S[qi[q], gi[copy_src[q]]] for q in o2s_ids if q in copy_src])
        b = np.mean([S[qi[q], gi[copy_src[q]]] for q in s2o_ids])
        if a < 1e-6 and b < 1e-6:
            xm = float(S[np.ix_(row_o2s, col_sar)].max())
            note = "掩码(0)；跨模态块 max %.3f" % xm
            ok &= xm > 0.5
        else:
            note = ""
            ok &= (a > 0.9 and b > 0.9)
        print("    sim  %-28s O2S %.3f | S2O %.3f  %s" % (nm, a, b, note))
    fnp = {}
    for nm, m in feats.items():
        qf = F.normalize(m["q"].float(), dim=1).numpy()
        gf = F.normalize(m["g"].float(), dim=1).numpy()
        fnp[nm] = (qf, gf)
        a = np.mean([np.dot(qf[qi[q]], gf[gi[copy_src[q]]]) for q in o2s_ids if q in copy_src])
        b = np.mean([np.dot(qf[qi[q]], gf[gi[copy_src[q]]]) for q in s2o_ids])
        print("    feat %-28s O2S %.3f | S2O %.3f" % (nm, a, b))
        ok &= (a > 0.5 and b > 0.5)
    if not ok:
        print("    !! 有通道对齐可疑")

    # ---------- 证据通道 ----------
    CH = []  # (src, group, dir, M)
    for nm, S in sims.items():
        CH.append((nm, "sim", "o2s", S[np.ix_(row_o2s, col_sar)]))
        CH.append((nm, "sim", "s2o", S[np.ix_(row_s2o, col_opt)].T))
    for nm, (qf, gf) in fnp.items():
        CH.append((nm, "fq", "o2s", qf[row_o2s] @ gf[col_sar].T))
        CH.append((nm, "fq", "s2o", (qf[row_s2o] @ gf[col_opt].T).T))
        CH.append((nm, "fg", "gg", gf[col_opt] @ gf[col_sar].T))
    ngrp = Counter(c[1] for c in CH)
    print("[C] 证据通道 %d 条: sim×%d（10模型×2向）| fq×%d（7×2）| fg×%d（7×1）" % (
        len(CH), ngrp["sim"], ngrp["fq"], ngrp["fg"]))

    RRF = []
    for src, grp, d, M in CH:
        flat = M.astype(np.float64).ravel()
        order = np.argsort(-flat, kind="stable")
        rank = np.empty(flat.size, dtype=np.float64)
        rank[order] = np.arange(flat.size, dtype=np.float64)
        RRF.append((1.0 / (K_RRF + rank)).reshape(M.shape))

    def fuse(sel, w=None):
        w = w or {}
        Acc = np.zeros((n, n))
        for idx in sel:
            Acc += w.get(CH[idx][1], 1.0) * RRF[idx]
        rr, cc = linear_sum_assignment(-Acc)
        pi = np.empty(n, dtype=int)
        pi[rr] = cc
        return Acc, pi

    def sel(group=None, dirs=None, srcs=None):
        out = []
        for idx, (src, grp, d, _) in enumerate(CH):
            if group and grp not in group:
                continue
            if dirs and d not in dirs:
                continue
            if srcs is not None and src not in srcs:
                continue
            out.append(idx)
        return out

    ALL = sel()
    F_all, pi_all = fuse(ALL)

    # ---------- 变体一致性（稳定性） ----------
    snames, fnames = sorted(sims), sorted(feats)
    halfA = sel(srcs=set(snames[0::2]) | set(fnames[0::2]))  # 5 sim + 4 feat
    halfB = sel(srcs=set(snames[1::2]) | set(fnames[1::2]))  # 5 sim + 3 feat
    VAR = {
        "sim-only(20)": fuse(sel(group={"sim"}))[1],
        "sim+fq(34)": fuse(sel(group={"sim", "fq"}))[1],
        "fq-only(14)": fuse(sel(group={"fq"}))[1],
        "fg-only(7)": fuse(sel(group={"fg"}))[1],
        "o2s-side(17)": fuse(sel(dirs={"o2s"}))[1],
        "s2o-side(17)": fuse(sel(dirs={"s2o"}))[1],
        "halfA(22)": fuse(halfA)[1],
        "halfB(19)": fuse(halfB)[1],
    }
    print("[D] 变体双射 vs 全量(41) 一致数（/120）:")
    for k, v in VAR.items():
        print("    %-14s %3d" % (k, int(np.sum(v == pi_all))))
    ab = int(np.sum(VAR["halfA(22)"] == VAR["halfB(19)"]))
    p_est = (ab / n) ** 0.5
    print("    halfA vs halfB : %3d  →  p ≈ sqrt(%.3f) = %.3f（两半独立证据的配对精度估计）" % (
        ab, ab / n, p_est))

    reid_srcs = (set(snames) - {nm for nm in snames if nm.startswith("sdfnet")}) | set(fnames)
    FAM = {
        "sdf-sim(6)": sel(group={"sim"}, srcs={nm for nm in snames if nm.startswith("sdfnet")}),
        "reid-sim(14)": sel(group={"sim"}, srcs={nm for nm in snames if not nm.startswith("sdfnet")}),
        "fq(14)": sel(group={"fq"}),
        "fg(7)": sel(group={"fg"}),
        "reid-all(35)": sel(srcs=reid_srcs),
        "all(41)": ALL,
    }
    pis = {k: fuse(v)[1] for k, v in FAM.items()}
    pis["all(41)"] = pi_all
    ks = list(FAM)
    print("[D2] 家族交叉一致数（/120）:")
    for a in range(len(ks)):
        row = " ".join("%s:%3d" % (ks[b].split("(")[0], int(np.sum(pis[ks[a]] == pis[ks[b]])))
                       for b in range(len(ks)) if b != a)
        print("    %-14s %s" % (ks[a], row))
    ag1 = int(np.sum(pis["sdf-sim(6)"] == pis["reid-sim(14)"]))
    ag2 = int(np.sum(pis["sdf-sim(6)"] == pis["fg(7)"]))
    print("    最独立对: sdf-sim vs reid-sim %d/120 → p≈%.3f | sdf-sim vs fg %d/120 → p≈%.3f" % (
        ag1, (ag1 / n) ** 0.5, ag2, (ag2 / n) ** 0.5))
    exact_mask = np.array([q in copy_src for q in o2s_ids])
    pS, pR = pis["sdf-sim(6)"], pis["reid-sim(14)"]
    print("    sdf vs reid 分列: 精确锚行 %d/93 | 近邻锚行 %d/27（近邻行更低=共同困难行，交叉估计偏乐观）" % (
        int(np.sum((pS == pR)[exact_mask])), int(np.sum((pS == pR)[~exact_mask]))))

    print("[E] 权重扫描（vs 全量一致数 /120）:")
    for tag, w in (("sim×2", {"sim": 2.0}), ("fq×2", {"fq": 2.0}), ("fg×2", {"fg": 2.0}),
                   ("sim×0.5", {"sim": 0.5}), ("fg×0.5", {"fg": 0.5})):
        print("    %-9s %3d" % (tag, int(np.sum(fuse(ALL, w)[1] == pi_all))))
    loo = ["%s=%d" % (nm, int(np.sum(fuse(sel(srcs=(set(snames) - {nm}) | set(fnames)))[1] == pi_all)))
           for nm in snames]
    print("[E2] 逐个剔除某 sim 模型后 vs all 一致数: " + "  ".join(loo))

    # ---------- 与基线 prediction 交叉验证 ----------
    base = Path(args.base) if args.base else Path("prediction.json")
    if not base.is_absolute():
        base = WS / base
    pred = json.loads(base.read_text(encoding="utf-8"))
    As2j = {A_s[j]: k for k, j in enumerate(s2o_ids)}
    Ao2i = {A_o[i]: k for k, i in enumerate(o2s_ids)}
    impl_j = [As2j.get(pred[q][0]) for q in o2s_ids]   # O2S i 的 top1 → S2O 列位置
    impl_i = [Ao2i.get(pred[q][0]) for q in s2o_ids]   # S2O j 的 top1 → O2S 行位置
    c_o2s = sum(x is not None for x in impl_j)
    c_s2o = sum(x is not None for x in impl_i)
    mutual = [(i, impl_j[i]) for i in range(n)
              if impl_j[i] is not None and impl_i[impl_j[i]] == i]
    ag_o2s = sum(1 for i in range(n) if impl_j[i] is not None and impl_j[i] == pi_all[i])
    ag_s2o = sum(1 for j in range(n) if impl_i[j] is not None and pi_all[impl_i[j]] == j)
    ag_mut = sum(1 for (i, j) in mutual if pi_all[i] == j)
    print("[F] 与基线 %s 一致性:" % base.name)
    print("    基线隐含落点: O2S top1∈A_s %d/120 | S2O top1∈A_o %d/120 | 互反闭环 %d 对" % (
        c_o2s, c_s2o, len(mutual)))
    print("    新双射与基线一致: O2S侧 %d/%d | S2O侧 %d/%d | 闭环对 %d/%d" % (
        ag_o2s, c_o2s, ag_s2o, c_s2o, ag_mut, len(mutual)))
    pS_ = pis["sdf-sim(6)"]
    flip_idx = [i for i in range(n) if impl_j[i] is not None and impl_j[i] != pi_all[i]]
    corr = sum(1 for i in flip_idx if pS_[i] == pi_all[i])
    sup_old = sum(1 for i in flip_idx if pS_[i] == impl_j[i])
    tri = sum(1 for i in range(n) if impl_j[i] is not None and impl_j[i] == pi_all[i] == pS_[i])
    none_idx = sum(1 for i in range(n) if impl_j[i] is None)
    print("[F2] 翻转行 %d 条（基线落点 vs 新π 不同）: sdf 家族支持新π %d | 支持基线 %d | 独立 %d" % (
        len(flip_idx), corr, sup_old, len(flip_idx) - corr - sup_old))
    print("    三方一致(基线=新π=sdf) %d 条 | 基线 O2S top1 不在 A_s 的 %d 条（基线必然错）" % (tri, none_idx))

    # ---------- 逐行置信度表（--diag；只读，不改动求解结果） ----------
    if args.diag:
        inv_all = np.empty(n, dtype=int)
        inv_all[pi_all] = np.arange(n)
        fams = ["sdf-sim(6)", "reid-sim(14)", "fq(14)", "fg(7)", "reid-all(35)"]
        var_ks = ["o2s-side(17)", "s2o-side(17)", "halfA(22)", "halfB(19)"]

        def stats(vec, pos):
            val = float(vec[pos])
            rank = int(np.sum(vec > val))
            margin = val - float(vec.max())
            return rank, margin

        hdr = ("side,idx,query,anchor_exact,pi_id,old_id,new_rank,new_margin,top1_id,"
               "n_sup_new,n_sup_old,n_var_new,n_var_old,sup_new,sup_old,var_new,var_old")
        lines, cand = [hdr], []
        for i in range(n):
            q = o2s_ids[i]
            new, old = int(pi_all[i]), impl_j[i]
            rank, margin = stats(F_all[i], new)
            top1 = s2o_ids[int(F_all[i].argmax())]
            sn = [k for k in fams if pis[k][i] == new]
            so = [k for k in fams if old is not None and pis[k][i] == old]
            vn = [k for k in var_ks if VAR[k][i] == new]
            vo = [k for k in var_ks if old is not None and VAR[k][i] == old]
            lines.append(",".join(map(str, [
                "O2S", i, q, int(q in copy_src), s2o_ids[new],
                s2o_ids[old] if old is not None else "-", rank, "%.6f" % margin, top1,
                len(sn), len(so), len(vn), len(vo),
                "|".join(sn), "|".join(so), "|".join(vn), "|".join(vo)])))
            if old is not None and old != new:
                cand.append(("O2S", i, q, s2o_ids[old], s2o_ids[new],
                             len(sn), len(so), len(vn), len(vo), margin))
        for j in range(n):
            q = s2o_ids[j]
            new, old = int(inv_all[j]), impl_i[j]
            rank, margin = stats(F_all[:, j], new)
            top1 = o2s_ids[int(F_all[:, j].argmax())]
            sn = [k for k in fams if pis[k][new] == j]
            so = [k for k in fams if old is not None and pis[k][old] == j]
            vn = [k for k in var_ks if VAR[k][new] == j]
            vo = [k for k in var_ks if old is not None and VAR[k][old] == j]
            lines.append(",".join(map(str, [
                "S2O", j, q, int(q in copy_src), o2s_ids[new],
                o2s_ids[old] if old is not None else "-", rank, "%.6f" % margin, top1,
                len(sn), len(so), len(vn), len(vo),
                "|".join(sn), "|".join(so), "|".join(vn), "|".join(vo)])))
            if old is not None and old != new:
                cand.append(("S2O", j, q, o2s_ids[old], o2s_ids[new],
                             len(sn), len(so), len(vn), len(vo), margin))
        outp_csv = PROJ / "outputs" / "pair_diag.csv"
        outp_csv.parent.mkdir(parents=True, exist_ok=True)
        outp_csv.write_text("\n".join(lines), encoding="utf-8")
        print("[K] 逐行置信度表 %d 行 → %s" % (len(lines) - 1, outp_csv))
        pos = sum(1 for t in cand if t[5] > t[6])
        neg = sum(1 for t in cand if t[5] < t[6])
        tie = len(cand) - pos - neg
        print("[K2] 冲突行 %d 条（基线 implied ≠ 新π）: 家族票 新>旧 %d | 新=旧 %d | 新<旧 %d" % (
            len(cand), pos, tie, neg))
        for t in sorted(cand, key=lambda x: (x[5] - x[6], x[9]))[:12]:
            print("     %-3s %s: sup 新%d/旧%d | var 新%d/旧%d | margin %+.5f | 旧列 %s → 新列 %s" % (
                t[0], t[2], t[5], t[6], t[7], t[8], t[9], t[3], t[4]))
        exact_i = [i for i in range(n) if o2s_ids[i] in copy_src]
        near_i = [i for i in range(n) if o2s_ids[i] not in copy_src]
        def hg(idxs):
            return sum(1 for i in idxs
                       if VAR["o2s-side(17)"][i] == VAR["s2o-side(17)"][i])
        print("[K3] 方向半分裂一致: 精确锚行 %d/%d | 近邻锚行 %d/%d" % (
            hg(exact_i), len(exact_i), hg(near_i), len(near_i)))

    # ---------- 软统计 ----------
    order_row = np.argsort(-F_all, axis=1)
    rank_of_pi = np.array([np.where(order_row[i] == pi_all[i])[0][0] for i in range(n)])
    rb, cb = F_all.argmax(axis=1), F_all.argmax(axis=0)
    print("[G] 软统计: 行argmax互异 %d/120 | 列argmax互异 %d/120 | π 行内 rank1 数 %d，≤3 数 %d" % (
        len(set(rb.tolist())), len(set(cb.tolist())),
        int((rank_of_pi == 0).sum()), int((rank_of_pi < 3).sum())))

    # ---------- O2O 身份重叠诊断 ----------
    row_o2o = np.array([qi[q] for q in o2o_ids])
    Acc = np.zeros((len(o2o_ids), n))
    for nm, S in sims.items():
        M2 = S[np.ix_(row_o2o, col_opt)]
        flat = M2.ravel()
        o2 = np.argsort(-flat, kind="stable")
        rk = np.empty(flat.size)
        rk[o2] = np.arange(flat.size)
        Acc += (1.0 / (K_RRF + rk)).reshape(M2.shape)
    for nm, (qf, gf) in fnp.items():
        M2 = qf[row_o2o] @ gf[col_opt].T
        flat = M2.ravel()
        o2 = np.argsort(-flat, kind="stable")
        rk = np.empty(flat.size)
        rk[o2] = np.arange(flat.size)
        Acc += (1.0 / (K_RRF + rk)).reshape(M2.shape)
    best = Acc.argmax(axis=1)
    top2 = np.sort(Acc, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    print("[H] O2O 诊断: 27 条对 A_o 的 argmax 互异 %d/27 | 边际 top1-top2 中位 %.4f/min %.4f" % (
        len(set(best.tolist())), float(np.median(margin)), float(margin.min())))
    o2o_fix = [k for k in range(len(o2o_ids)) if pred[o2o_ids[k]][0] != A_o[o2s_ids[int(best[k])]]]
    print("[H2] O2O 修复候选（基线 top1 ≠ 身份 argmax 锚）: %d/27" % len(o2o_fix))
    for k in o2o_fix:
        r = int(np.sum(Acc[k] > Acc[k][best[k]]))
        print("     %s: 基线 %s → argmax %s（列内 rank %d, 边际 %+.4f）" % (
            o2o_ids[k], pred[o2o_ids[k]][0], A_o[o2s_ids[int(best[k])]], r, float(margin[k])))

    p_lo = (min(ag1, ag2) / n) ** 0.5  # 跨家族独立口径（保守）
    p_hi = p_est                        # 半分裂口径（乐观，同池相关性抬高）
    print("[I] 预估: p ∈ [%.3f, %.3f] → Final ≈ 0.9·p + 0.095 ∈ [%.3f, %.3f]（O2O 沿用现役 ~0.95）" % (
        p_lo, p_hi, 0.9 * p_lo + 0.095, 0.9 * p_hi + 0.095))

    # ---------- 合成（可选） ----------
    if args.emit:
        pi = pi_all
        inv = np.empty(n, dtype=int)
        inv[pi] = np.arange(n)
        new, flips = {}, [0, 0]
        for i, q in enumerate(o2s_ids):
            r1 = A_s[s2o_ids[pi[i]]]
            new[q] = ([r1] + [x for x in pred[q] if x != r1])[:10]
            flips[0] += int(pred[q][0] != r1)
        for j, q in enumerate(s2o_ids):
            r1 = A_o[o2s_ids[inv[j]]]
            new[q] = ([r1] + [x for x in pred[q] if x != r1])[:10]
            flips[1] += int(pred[q][0] != r1)
        for q in o2o_ids:
            new[q] = list(pred[q])
        for q, lst in new.items():
            assert len(lst) == 10 and len(set(lst)) == 10, q
            want = "sar" if q in set(o2s_ids) else "optical"
            assert all(gmod[x] == want for x in lst), q
        outp = Path(args.out) if args.out else Path("prediction_closedpair.json")
        if not outp.is_absolute():
            outp = WS / outp
        outp.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[J] 已写出 %s（O2S top1 翻转 %d/120，S2O 翻转 %d/120；O2O 原样）" % (
            outp, flips[0], flips[1]))
        for tag, ids in (("O2S", o2s_ids), ("S2O", s2o_ids), ("O2O", o2o_ids)):
            inter = np.mean([len(set(new[q]) & set(pred[q])) for q in ids])
            same = sum(1 for q in ids if new[q] == pred[q])
            print("    %s: 与基线 top-10 平均重合 %.2f/10，完全一致 %d/%d" % (
                tag, inter, same, len(ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())