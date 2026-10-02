"""推理后处理：k-reciprocal 重排序 + 查询扩展（QE）。

均不修改模型权重，仅在特征/相似度层面操作，可叠加使用。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# k-reciprocal Re-ranking
# --------------------------------------------------------------------------- #
def k_reciprocal_rerank(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> torch.Tensor:
    """经典 k-reciprocal 重排序（Zhong et al., CVPR 2017）。

    Args:
        query_feats: (Q, D) L2 归一化特征
        gallery_feats: (G, D) L2 归一化特征
        k1: 候选近邻数（reciprocal 判定用）
        k2: Jaccard 平滑的局部近邻数
        lambda_value: 原始相似度与重排序相似度的融合权重

    Returns:
        reranked_sim: (Q, G) 重排序后的相似度矩阵
    """
    q = query_feats
    g = gallery_feats
    Q, G = q.size(0), g.size(0)
    device = q.device

    # 原始余弦相似度
    sim_qg = torch.matmul(q, g.t())  # (Q, G)
    # gallery 内部相似度
    sim_gg = torch.matmul(g, g.t())  # (G, G)

    # 对每个 gallery 找其 top-k1 近邻（含自身）
    # forward: 对 query，其 top-k1 gallery 候选
    # backward: 对每个 gallery，其在 gallery 中的 top-k1 近邻
    k1 = min(k1, G)

    # gallery 的近邻集合
    gg_topk_idx = sim_gg.topk(k1, dim=1).indices  # (G, k1)
    # 用稀疏矩阵表示 V_g: V_g[i, j] = 1 当 j 是 i 的 k1 近邻
    V_g = torch.zeros(G, G, device=device, dtype=torch.float32)
    rows = torch.arange(G, device=device).unsqueeze(1).expand(-1, k1)
    V_g[rows, gg_topk_idx] = 1.0

    # 对每个 query，初始 top-k1 候选
    qg_topk_idx = sim_qg.topk(k1, dim=1).indices  # (Q, k1)

    # 计算重排序相似度（按 query 逐个处理，避免显存爆炸）
    reranked = torch.zeros(Q, G, device=device, dtype=torch.float32)
    for qi in range(Q):
        cand_idx = qg_topk_idx[qi]  # (k1,)
        # query 的候选集合 V_q
        V_q = torch.zeros(G, device=device, dtype=torch.float32)
        V_q[cand_idx] = 1.0
        # 对每个候选 c，其 gallery 近邻集合 V_g[c]
        # Jaccard(q, c) = |V_q ∩ V_g[c]| / |V_q ∪ V_g[c]|
        # 向量化：对所有候选
        cand_V_g = V_g[cand_idx]  # (k1, G)
        intersection = (V_q.unsqueeze(0) * cand_V_g).sum(dim=1)  # (k1,)
        union = (V_q.unsqueeze(0) + cand_V_g).clamp(max=1.0).sum(dim=1)  # (k1,)
        jaccard = intersection / union.clamp(min=1e-6)  # (k1,)
        reranked[qi, cand_idx] = jaccard

    # 融合：final = lambda * reranked + (1 - lambda) * original
    final = lambda_value * reranked + (1.0 - lambda_value) * sim_qg
    return final


# --------------------------------------------------------------------------- #
# Query Expansion (QE)
# --------------------------------------------------------------------------- #
def query_expansion(
    query_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    topk: int = 1,
    alpha: float = 0.5,
) -> torch.Tensor:
    """用 top-k gallery 特征扩展 query 特征。

    q_new = normalize(q + alpha * mean(g_topk))

    Args:
        query_feats: (Q, D) L2 归一化
        gallery_feats: (G, D) L2 归一化
        topk: 取前 k 个 gallery 做扩展
        alpha: 扩展权重

    Returns:
        expanded: (Q, D) L2 归一化的扩展 query 特征
    """
    sim = torch.matmul(query_feats, gallery_feats.t())  # (Q, G)
    topk = min(topk, gallery_feats.size(0))
    topk_idx = sim.topk(topk, dim=1).indices  # (Q, topk)
    # 取 topk gallery 特征并平均
    topk_feats = gallery_feats[topk_idx]  # (Q, topk, D)
    avg_g = topk_feats.mean(dim=1)  # (Q, D)
    expanded = F.normalize(query_feats + alpha * avg_g, dim=1)
    return expanded
