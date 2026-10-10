"""三元组损失（含难样本挖掘变体）。

- TripletLoss: 经典 batch-hard 三元组（欧氏距离）
- HardMiningTripletLoss: 难样本挖掘版，提供 margin 与自适应 margin 选项
- cross_modal_hard_triplet: 双向跨模态 hard mining 函数（O→S 与 S→O）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-Hard 三元组损失，欧氏距离。"""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(features, features, p=2)  # (N, N)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        eye = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
        mask = mask & ~eye

        # 每个 anchor 的最近负样本 / 最远正样本
        max_positive = torch.where(mask, dist, torch.zeros_like(dist)).amax(dim=1)
        mask_neg = ~mask & ~eye
        # 负样本距离：将正样本置为 -inf
        dist_neg = torch.where(mask_neg, dist, torch.full_like(dist, -1e9))
        min_negative = dist_neg.amin(dim=1)

        loss = F.relu(max_positive - min_negative + self.margin)
        valid = mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        return loss[valid].mean()


class HardMiningTripletLoss(nn.Module):
    """显式难样本挖掘版三元组损失：每 batch 内取难正/难负对。"""

    def __init__(self, margin: float = 0.3, adaptive_margin: bool = False) -> None:
        super().__init__()
        self.margin = margin
        self.adaptive = adaptive_margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(features, features, p=2)
        n = features.size(0)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(n, device=features.device, dtype=torch.bool)
        mask = mask & ~eye
        mask_neg = ~mask & ~eye

        pos_dist = torch.where(mask, dist, torch.zeros_like(dist))
        hard_pos = pos_dist.amax(dim=1)
        neg_dist = torch.where(mask_neg, dist, torch.full_like(dist, 1e9))
        hard_neg = neg_dist.amin(dim=1)

        margin = self.margin
        if self.adaptive:
            # 自适应 margin：按样本对难度动态缩放
            margin = self.margin + 0.1 * torch.sigmoid(hard_pos - hard_neg)

        loss = F.relu(hard_pos - hard_neg + margin)
        valid = mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        return loss[valid].mean()


def cross_modal_hard_triplet(
    features: torch.Tensor,
    labels: torch.Tensor,
    modalities: torch.Tensor,
    margin: float = 0.3,
    w_os: float = 1.0,
    w_so: float = 1.0,
) -> torch.Tensor:
    """双向跨模态 hard mining 三元组损失（可复用函数）。

    O→S 方向：anchor=光学样本，positive=同身份 SAR 中距离最近者，
              negative=不同身份 SAR 中距离最近者；
    S→O 方向对称构造。某身份缺另一模态样本时该 anchor 跳过。
    两个方向各自求均值后按「有效方向权重」加权平均。

    w_os / w_so：方向权重，用于按竞赛方向非对称强调（S2O = SAR query → 光学 gallery，
    对应 anchor=SAR 的 S→O 方向）。默认 1.0/1.0 与旧行为逐位一致。
    """
    device = features.device
    o_mask = modalities == 0
    s_mask = modalities == 1
    o_feat, o_lab = features[o_mask], labels[o_mask]
    s_feat, s_lab = features[s_mask], labels[s_mask]
    if o_feat.size(0) == 0 or s_feat.size(0) == 0:
        return features.new_tensor(0.0)

    dist = torch.cdist(o_feat, s_feat, p=2)  # (No, Ns)
    same = o_lab.unsqueeze(1) == s_lab.unsqueeze(0)  # (No, Ns)

    big = torch.full_like(dist, 1e9)
    # O→S
    hard_pos_os = torch.where(same, dist, big).amin(dim=1)
    hard_neg_os = torch.where(~same, dist, big).amin(dim=1)
    valid_os = (same.sum(dim=1) > 0) & (~same).any(dim=1)
    if valid_os.sum() > 0:
        loss_os = F.relu(hard_pos_os - hard_neg_os + margin)[valid_os].mean()
    else:
        loss_os = features.new_tensor(0.0)

    # S→O
    dist_so = dist.t()
    same_so = same.t()
    hard_pos_so = torch.where(same_so, dist_so, big.t()).amin(dim=1)
    hard_neg_so = torch.where(~same_so, dist_so, big.t()).amin(dim=1)
    valid_so = (same_so.sum(dim=1) > 0) & (~same_so).any(dim=1)
    if valid_so.sum() > 0:
        loss_so = F.relu(hard_pos_so - hard_neg_so + margin)[valid_so].mean()
    else:
        loss_so = features.new_tensor(0.0)

    n_dir = (valid_os.sum() > 0).float() * w_os + (valid_so.sum() > 0).float() * w_so
    if n_dir == 0:
        return features.new_tensor(0.0)
    return (w_os * loss_os + w_so * loss_so) / n_dir
