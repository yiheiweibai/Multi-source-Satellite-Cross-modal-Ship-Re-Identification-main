"""损失组合：跨模态 SupCon + 三元组 + 身份分类，按权重加权求和。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .supcon_loss import SupConLoss
from .triplet_loss import HardMiningTripletLoss


class ComposedLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.supcon = SupConLoss(temperature=getattr(cfg, "supcon_temperature", 0.07))
        self.triplet = HardMiningTripletLoss(
            margin=getattr(cfg, "triplet_margin", 0.3),
            adaptive_margin=getattr(cfg, "adaptive_margin", False),
        )
        self.w_supcon = getattr(cfg, "w_supcon", 0.5)
        self.w_triplet = getattr(cfg, "w_triplet", 0.3)
        self.w_ce = getattr(cfg, "w_ce", 1.0)
        self.w_arcface = getattr(cfg, "w_arcface", 0.0)
        self.label_smooth = getattr(cfg, "label_smooth", 0.1)

    def forward(
        self,
        ret_feat: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        arcface_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回 dict: {loss, supcon, triplet, ce, arcface}。"""
        loss_supcon = self.supcon(ret_feat, labels)
        loss_triplet = self.triplet(ret_feat, labels)

        # 身份分类（标签平滑）
        n_classes = logits.size(1)
        if n_classes > 1:
            targets = torch.full(
                (labels.size(0), n_classes), self.label_smooth / (n_classes - 1), device=logits.device
            )
            targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smooth)
            loss_ce = F.cross_entropy(logits, targets)
        else:
            loss_ce = F.cross_entropy(logits, labels)

        total = (
            self.w_supcon * loss_supcon
            + self.w_triplet * loss_triplet
            + self.w_ce * loss_ce
        )

        loss_arcface = torch.tensor(0.0, device=logits.device)
        if self.w_arcface > 0 and arcface_logits is not None:
            loss_arcface = F.cross_entropy(arcface_logits, labels)
            total = total + self.w_arcface * loss_arcface

        return {
            "loss": total,
            "supcon": loss_supcon,
            "triplet": loss_triplet,
            "ce": loss_ce,
            "arcface": loss_arcface,
        }
