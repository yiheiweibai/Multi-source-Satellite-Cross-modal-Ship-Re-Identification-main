# -*- coding: utf-8 -*-
"""Phase 5：学习型排序融合（Learning-to-Rank Fusion）。

替代 RRF 的手工排名融合，训练一个轻量级排序模型（sklearn GradientBoostingRegressor 作为
pointwise ranker），在 local_val 上学习各成员的分数/排名特征 → 候选相关性的映射。

特征工程（每个 query-gallery-candidate 一个样本）：
  - 每个成员的 z-score 归一化相似度分数（连续值，保留量纲信息）
  - 每个成员的倒数排名 1/(k+rank)（RRF 特征，保留序数信息）
  - 每个成员在该 query 候选中的排名（原始整数排名）
  - 候选被多少成员选入其 top-100（成员投票数，鲁棒性特征）

标签：该候选是否为该 query 的正确匹配（0/1）。

训练/评估协议（嵌套交叉验证，避免 local_val 过拟合）：
  - fold0 训练 → fold1 评估；fold1 训练 → fold0 评估
  - 两折最终得分均值 ≥ RRF 基线 + 0.003 才采用

用法：
  python scripts/rank_fusion.py --eval-only   # 只做交叉验证评估
  python scripts/rank_fusion.py --train --out models/rank_fusion.pkl  # 训练并保存模型
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
DIRECTIONS = ("O2S", "S2O", "O2O")
WEIGHTS = {"O2S": 0.45, "S2O": 0.45, "O2O": 0.10}
TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}
MEMBER_TOPK = 100  # 每个成员提供的候选深度
RRF_K = 10.0       # RRF 特征常数
POOL_SIZE = 50     # 候选池大小（控制训练样本数与推理成本）

# 成员 sim 矩阵路径 + 权重（与 fusion_compare.py 一致，保持公平对比）
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
MEMBER_NAMES = list(MEMBERS.keys())

TASK = ROOT.parent / "question6-data" / "traindata" / "local_val_task.json"
GT   = ROOT.parent / "question6-data" / "traindata" / "local_val_gt.json"


def load_task():
    with open(TASK, encoding="utf-8") as f:
        return json.load(f)


def load_gt():
    with open(GT, encoding="utf-8") as f:
        gt = json.load(f)
    return {k: set(v) if isinstance(v, list) else set([v]) for k, v in gt.items()}


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


def build_candidate_pool(sims, masks, queries, gallery, pool_size=POOL_SIZE):
    """用 RRF(k=10) 构建每个 query 的候选池（确定性，避免随机性泄漏）。"""
    member_weights = {n: w for n, (_, w) in MEMBERS.items()}
    pools = []
    for qi, q in enumerate(queries):
        cand = masks[qi]
        cand_t = torch.tensor(cand, dtype=torch.long)
        scores = {}
        for name, sim in sims.items():
            w = member_weights[name]
            vals = sim[qi, cand_t]
            order = torch.argsort(vals, descending=True).tolist()
            for rank, idx in enumerate(order[:MEMBER_TOPK], 1):
                gid = gallery[cand[idx]]["image_id"]
                scores[gid] = scores.get(gid, 0.0) + w / (RRF_K + rank)
        pool = [gid for gid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:pool_size]]
        pools.append(pool)
    return pools


def extract_features(sims, masks, queries, gallery, pools):
    """为每个 query 的候选池构建特征矩阵 X 和标签 y。

    特征维度：len(MEMBERS) * 3 + 1
      - 每成员：z-score 分数、RRF 特征 1/(k+rank)、原始排名（top-MEMBER_TOPK 内，否则为 MEMBER_TOPK+1）
      - 最后一维：成员投票数（该候选被多少成员选入 top-MEMBER_TOPK）
    """
    member_weights = {n: w for n, (_, w) in MEMBERS.items()}
    gid_to_idx = {g["image_id"]: i for i, g in enumerate(gallery)}

    X_list, y_list, qid_list, gid_list = [], [], [], []
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        cand = masks[qi]
        cand_t = torch.tensor(cand, dtype=torch.long)

        # 预计算每成员对该 query 候选的分数、排名、RRF 特征
        member_data = {}
        for name, sim in sims.items():
            w = member_weights[name]
            vals = sim[qi, cand_t].float()
            order = torch.argsort(vals, descending=True).tolist()
            rank_map = {}   # gallery_idx -> (score, rank)
            for rank, idx in enumerate(order[:MEMBER_TOPK], 1):
                rank_map[cand[idx]] = (vals[idx].item(), rank)
            member_data[name] = (w, rank_map)

        for gid in pools[qi]:
            gidx = gid_to_idx[gid]
            feats = []
            votes = 0
            for name in MEMBER_NAMES:
                w, rank_map = member_data[name]
                if gidx in rank_map:
                    score, rank = rank_map[gidx]
                    feats.extend([score, w / (RRF_K + rank), float(rank)])
                    votes += 1
                else:
                    feats.extend([0.0, 0.0, float(MEMBER_TOPK + 1)])
            feats.append(float(votes))
            X_list.append(feats)
            y_list.append(0.0)  # 标签在调用方填充
            qid_list.append(qid)
            gid_list.append(gid)

    X = np.array(X_list, dtype=np.float32)
    return X, qid_list, gid_list


def fill_labels(y_list_placeholder, qid_list, gid_list, gt):
    y = np.array([1.0 if gid in gt.get(qid, set()) else 0.0
                  for qid, gid in zip(qid_list, gid_list)], dtype=np.float32)
    return y


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


def fold_split(queries):
    """按 query_type 内部交替切分 folds[i % 2]（与 fusion_compare.py 一致）。"""
    folds = {0: [], 1: []}
    for d in DIRECTIONS:
        sub = [q for q in queries if q["query_type"] == d]
        for i, q in enumerate(sub):
            folds[i % 2].append(q)
    return folds


def rrf_baseline(sims, masks, queries, gallery, k=RRF_K, topk=10):
    """RRF 基线（与 fusion_compare.py 完全一致）。"""
    member_weights = {n: w for n, (_, w) in MEMBERS.items()}
    out = {}
    for qi, q in enumerate(queries):
        cand = masks[qi]
        scores = {}
        for name, sim in sims.items():
            w = member_weights[name]
            vals = sim[qi, torch.tensor(cand, dtype=torch.long)]
            order = torch.argsort(vals, descending=True).tolist()
            for rank, idx in enumerate(order[:MEMBER_TOPK], 1):
                gid = gallery[cand[idx]]["image_id"]
                scores[gid] = scores.get(gid, 0.0) + w / (k + rank)
        out[q["query_id"]] = [gid for gid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def train_ranker(X_train, y_train):
    """训练 pointwise 排序模型（GradientBoostingRegressor，轻量且对小样本稳健）。"""
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_ranker(model, sims, masks, queries, gallery, pools, topk=10):
    """用训练好的排序模型对候选池重排，输出 top-k 预测。"""
    X, qid_list, gid_list = extract_features(sims, masks, queries, gallery, pools)
    scores = model.predict(X)

    out = {}
    for qid, gid, s in zip(qid_list, gid_list, scores):
        out.setdefault(qid, []).append((gid, float(s)))
    for qid in out:
        out[qid] = [gid for gid, _ in sorted(out[qid], key=lambda kv: -kv[1])[:topk]]
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 5 学习型排序融合")
    ap.add_argument("--eval-only", action="store_true", help="只做嵌套交叉验证评估，不保存模型")
    ap.add_argument("--train", action="store_true", help="在全量 local_val 上训练并保存模型")
    ap.add_argument("--out", type=str, default="models/rank_fusion.pkl", help="模型保存路径")
    args = ap.parse_args()

    task = load_task()
    queries, gallery = task["queries"], task["gallery"]
    gt = load_gt()
    folds = fold_split(queries)

    print(f"成员数: {len(MEMBERS)} | queries: {len(queries)} | gallery: {len(gallery)}")
    print("加载 sim 矩阵...")
    sims = load_sims(queries, gallery)
    masks = mod_mask(queries, gallery)
    print("构建 RRF 候选池...")
    pools = build_candidate_pool(sims, masks, queries, gallery)

    # RRF 基线（用于对比）
    print("\n" + "=" * 70)
    print("RRF 基线（k=10, member_topk=100）")
    print("=" * 70)
    pred_rrf = rrf_baseline(sims, masks, queries, gallery)
    res_rrf = evaluate(pred_rrf, gt, queries)
    print(f"  全量 Final={res_rrf['Final']:.4f} | "
          f"O2S={res_rrf['O2S']['score']:.4f} S2O={res_rrf['S2O']['score']:.4f} O2O={res_rrf['O2O']['score']:.4f}")
    for f in (0, 1):
        sub = folds[f]
        sub_pred = {q["query_id"]: pred_rrf[q["query_id"]] for q in sub}
        res_f = evaluate(sub_pred, gt, sub)
        print(f"  fold{f} Final={res_f['Final']:.4f}")

    # 嵌套交叉验证评估学习型排序
    print("\n" + "=" * 70)
    print("学习型排序（嵌套交叉验证：fold0 训练 → fold1 评估；fold1 训练 → fold0 评估）")
    print("=" * 70)

    fold_finals = []
    for train_fold, eval_fold in ((0, 1), (1, 0)):
        train_queries = folds[train_fold]
        eval_queries = folds[eval_fold]
        train_qids = {q["query_id"] for q in train_queries}

        # 构建训练集（仅用 train_fold 的 query）
        train_indices = [i for i, q in enumerate(queries) if q["query_id"] in train_qids]
        X_all, qid_list, gid_list = extract_features(sims, masks, queries, gallery, pools)
        y_all = fill_labels(None, qid_list, gid_list, gt)

        train_mask = np.array([qid in train_qids for qid in qid_list])
        X_train, y_train = X_all[train_mask], y_all[train_mask]

        print(f"\nfold{train_fold} 训练样本数: {len(y_train)} (正样本 {int(y_train.sum())})")
        model = train_ranker(X_train, y_train)

        # 在 eval_fold 上评估
        eval_set = set(q["query_id"] for q in eval_queries)
        pred_eval = predict_with_ranker(model, sims, masks, queries, gallery, pools)
        pred_eval = {qid: pred_eval[qid] for qid in eval_set}
        res_eval = evaluate(pred_eval, gt, eval_queries)
        fold_finals.append(res_eval["Final"])
        print(f"fold{train_fold} 训练 → fold{eval_fold} 评估: Final={res_eval['Final']:.4f} | "
              f"O2S={res_eval['O2S']['score']:.4f} S2O={res_eval['S2O']['score']:.4f} O2O={res_eval['O2O']['score']:.4f}")

    avg_final = sum(fold_finals) / len(fold_finals)
    print(f"\n两折平均 Final={avg_final:.4f} | RRF 基线 fold 平均="
          f"{(evaluate({q['query_id']: pred_rrf[q['query_id']] for q in folds[0]}, gt, folds[0])['Final'] + evaluate({q['query_id']: pred_rrf[q['query_id']] for q in folds[1]}, gt, folds[1])['Final']) / 2:.4f}")

    # 全量训练并保存模型（用于测试集推理）
    if args.train:
        print("\n" + "=" * 70)
        print("在全量 local_val 上训练最终模型")
        print("=" * 70)
        X_all, qid_list, gid_list = extract_features(sims, masks, queries, gallery, pools)
        y_all = fill_labels(None, qid_list, gid_list, gt)
        print(f"训练样本数: {len(y_all)} (正样本 {int(y_all.sum())})")
        final_model = train_ranker(X_all, y_all)

        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "member_names": MEMBER_NAMES,
                "member_weights": {n: w for n, (_, w) in MEMBERS.items()},
                "pool_size": POOL_SIZE,
                "rrf_k": RRF_K,
                "member_topk": MEMBER_TOPK,
            }, f)
        print(f"模型已保存: {out_path}")

        # 全量 self-eval（仅作 sanity check，存在过拟合，不作为采纳依据）
        pred_full = predict_with_ranker(final_model, sims, masks, queries, gallery, pools)
        res_full = evaluate(pred_full, gt, queries)
        print(f"全量 self-eval（过拟合，仅供参考）: Final={res_full['Final']:.4f}")


if __name__ == "__main__":
    main()
