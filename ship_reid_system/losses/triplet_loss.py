"""三元组损失（含难样本挖掘变体）。

- TripletLoss: 经典 batch-hard 三元组（欧氏距离）
- HardMiningTripletLoss: 难样本挖掘版，提供 margin 与自适应 margin 选项
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
