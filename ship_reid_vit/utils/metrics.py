"""评测指标：mAP 与 Recall@K（支持跨模态/同模态子集）。

约定：gallery_labels 可为 1D (G,)（整体评估）或 2D (Q, Gmax)（子集评估，
每查询候选数量可能不同，用 -1 填充无效列）。
"""
from __future__ import annotations

from typing import Sequence

import torch


def _row_labels(gallery_labels: torch.Tensor, i: int) -> torch.Tensor:
    """取出第 i 行候选标签；1D 输入按整体图库返回。"""
    if gallery_labels.dim() == 1:
        return gallery_labels
    return gallery_labels[i]


def mean_average_precision(
    sim: torch.Tensor, labels: torch.Tensor, gallery_labels: torch.Tensor
) -> float:
    """输入为相似度矩阵 (Q, G)（越大越相似），返回整体 mAP。"""
    q = sim.size(0)
    ap_sum, count = 0.0, 0
    for i in range(q):
        gl = _row_labels(gallery_labels, i)
        order = sim[i].argsort(descending=True)
        valid = gl[order] != -1
        rel = ((labels[i] == gl[order]) & valid).float()
        if rel.sum() == 0:
            continue
        pos_rank = torch.nonzero(rel == 1).squeeze(1) + 1  # 1-based rank
        cum = torch.cumsum(rel, dim=0)
        prec = cum[pos_rank - 1] / pos_rank
        ap_sum += prec.mean().item()
        count += 1
    return ap_sum / max(count, 1)


def recall_at_k(
    sim: torch.Tensor, labels: torch.Tensor, gallery_labels: torch.Tensor, k: int
) -> float:
    q = sim.size(0)
    hits = 0
    for i in range(q):
        gl = _row_labels(gallery_labels, i)
        order = sim[i].argsort(descending=True)[:k]
        if (labels[i] == gl[order]).any():
            hits += 1
    return hits / max(q, 1)


def evaluate_retrieval(
    feats: torch.Tensor,
    labels: torch.Tensor,
    modalities: torch.Tensor,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict:
    """全量评估：整体 + 同模态 + 跨模态子集。

    Args:
        feats: (N, D) L2 归一化特征
        labels: (N,) 身份标签
        modalities: (N,) 0=optical 1=sar
    """
    sim = feats @ feats.t()
    out: dict = {"mAP": 0.0}
    for k in k_values:
        out[f"R{k}"] = 0.0

    # 整体：每行查询在全部样本中检索（候选标签 2D 广播，便于统一处理）
    # 排除查询自身（对角线），避免自匹配造成指标虚高
    sim_eval = sim.clone()
    sim_eval.fill_diagonal_(float("-inf"))
    gallery_all = labels.unsqueeze(0).expand(labels.size(0), labels.size(0))
    out["mAP"] = mean_average_precision(sim_eval, labels, gallery_all)
    for k in k_values:
        out[f"R{k}"] = recall_at_k(sim_eval, labels, gallery_all, k)

    # 子集：同模态（查询与候选模态一致，排除自身）与跨模态（模态不一致）
    for name, same_modal in (("same", True), ("cross", False)):
        q_ids, g_ids = [], []
        for i in range(sim.size(0)):
            if same_modal:
                cand = (modalities == modalities[i]).nonzero().squeeze(1)
                cand = cand[cand != i]  # 同模态检索排除查询自身
            else:
                cand = (modalities != modalities[i]).nonzero().squeeze(1)
            if cand.numel() == 0:
                continue
            q_ids.append(i)
            g_ids.append(cand)
        if not q_ids:
            continue
        # 构造子集相似度矩阵：每查询行 -> 该查询的候选列（不足补 -1）
        max_g = max(len(c) for c in g_ids)
        sub_sim = torch.full((len(q_ids), max_g), float("-inf"))
        sub_gallery_labels = torch.full((len(q_ids), max_g), -1, dtype=torch.long)
        for r, (qi, cand) in enumerate(zip(q_ids, g_ids)):
            sub_sim[r, : len(cand)] = sim[qi, cand]
            sub_gallery_labels[r, : len(cand)] = labels[cand]
        sub_labels = labels[torch.tensor(q_ids)]
        out[f"{name}_mAP"] = mean_average_precision(sub_sim, sub_labels, sub_gallery_labels)
        for k in k_values:
            out[f"{name}_R{k}"] = recall_at_k(sub_sim, sub_labels, sub_gallery_labels, k)
    return out


# ============ 赛题提交结果评测（O2S / S2O / O2O） ============

def map_at_k(hit_flags: Sequence[int], num_rel: int, k: int = 10) -> float:
    """AP@K：hit_flags 为长度为 K 的 0/1 命中序列，按排序位置计算。

    AP = Σ(P@k * rel_k) / min(num_rel, K)；无正样本时返回 0。
    """
    num_rel = int(num_rel)
    if num_rel <= 0:
        return 0.0
    k = min(int(k), len(hit_flags))
    if k <= 0:
        return 0.0
    hits = 0
    ap = 0.0
    for pos in range(k):
        if hit_flags[pos]:
            hits += 1
            ap += hits / (pos + 1)
    return ap / min(num_rel, k)


def submission_direction_metrics(
    prediction: dict,
    gt_positives: dict,
    queries: Sequence[dict],
    direction: str,
    topk: int = 10,
) -> dict:
    """计算单个检索方向（O2S/S2O/O2O）的 R@1、mAP@10 与方向得分。

    Args:
        prediction: {query_id: [gallery image_id, ...]}（按相似度降序）
        gt_positives: {query_id: set(gallery image_id)} 正确候选集合
        queries: task.json 的 queries 列表（含 query_id / query_type）
        direction: "O2S" / "S2O" / "O2O"
    """
    sub = [q for q in queries if q.get("query_type") == direction]
    n = max(len(sub), 1)
    r1_hits = 0
    ap_sum = 0.0
    for q in sub:
        cands = prediction.get(q["query_id"], [])[:topk]
        pos = gt_positives.get(q["query_id"], set())
        hits = [1 if c in pos else 0 for c in cands]
        if hits and hits[0] == 1:
            r1_hits += 1
        ap_sum += map_at_k(hits, len(pos), k=topk)
    r1 = r1_hits / n
    mAP10 = ap_sum / n
    return {"count": len(sub), "R@1": r1, "mAP@10": mAP10, "score": (r1 + mAP10) / 2}


def evaluate_submission(
    prediction: dict,
    gt_positives: dict,
    queries: Sequence[dict],
    topk: int = 10,
    weights: Sequence[float] = (0.45, 0.45, 0.10),
) -> dict:
    """赛题综合评测：三个方向分别 R@1 与 mAP@10，加权综合。

    综合得分 = w_O2S * score_O2S + w_S2O * score_S2O + w_O2O * score_O2O。
    """
    directions = ("O2S", "S2O", "O2O")
    result = {}
    for d, w in zip(directions, weights):
        m = submission_direction_metrics(prediction, gt_positives, queries, d, topk=topk)
        m["weight"] = w
        result[d] = m
    overall = sum(result[d]["score"] * result[d]["weight"] for d in directions)
    result["overall"] = overall
    return result
