"""损失组合：跨模态 SupCon + 三元组 + 身份分类 + 可选跨模态损失，按权重加权求和。

实验开关（全部默认关闭，保证旧配置零改动可复现）：
- w_cm_infonce / w_cm_triplet: 跨模态 InfoNCE / hard triplet 权重（>0 时启用）
- cm_temperature / cm_margin: 对应温度与 margin
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .supcon_loss import SupConLoss
from .triplet_loss import HardMiningTripletLoss
from .cross_modal_loss import CrossModalInfoNCE, CrossModalHardTriplet


class ComposedLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.supcon = SupConLoss(temperature=getattr(cfg, "supcon_temperature", 0.07))
        self.triplet = HardMiningTripletLoss(
            margin=getattr(cfg, "triplet_margin", 0.3),
            adaptive_margin=getattr(cfg, "adaptive_margin", False),
        )
        self.cm_infonce = CrossModalInfoNCE(temperature=getattr(cfg, "cm_temperature", 0.07))
        self.cm_triplet = CrossModalHardTriplet(margin=getattr(cfg, "cm_margin", 0.3))
        self.w_supcon = getattr(cfg, "w_supcon", 0.5)
        self.w_triplet = getattr(cfg, "w_triplet", 0.3)
        self.w_ce = getattr(cfg, "w_ce", 1.0)
        self.w_arcface = getattr(cfg, "w_arcface", 0.0)
        self.w_cm_infonce = getattr(cfg, "w_cm_infonce", 0.0)
        self.w_cm_triplet = getattr(cfg, "w_cm_triplet", 0.0)
        self.label_smooth = getattr(cfg, "label_smooth", 0.1)

    def forward(
        self,
        ret_feat: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modalities: torch.Tensor | None = None,
        arcface_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回 dict: {loss, supcon, triplet, ce, arcface, cm_infonce, cm_triplet}。"""
        # AMP 兼容：loss 统一在 fp32 下计算，规避 autocast 下半精度数值不稳
        ret_feat = ret_feat.float()
        logits = logits.float()
        if arcface_logits is not None:
            arcface_logits = arcface_logits.float()

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

        # 跨模态损失（默认权重 0，不启用）
        loss_cm_infonce = torch.tensor(0.0, device=logits.device)
        loss_cm_triplet = torch.tensor(0.0, device=logits.device)
        if modalities is not None and self.w_cm_infonce > 0:
            loss_cm_infonce = self.cm_infonce(ret_feat, labels, modalities)
            total = total + self.w_cm_infonce * loss_cm_infonce
        if modalities is not None and self.w_cm_triplet > 0:
            loss_cm_triplet = self.cm_triplet(ret_feat, labels, modalities)
            total = total + self.w_cm_triplet * loss_cm_triplet

        return {
            "loss": total,
            "supcon": loss_supcon,
            "triplet": loss_triplet,
            "ce": loss_ce,
            "arcface": loss_arcface,
            "cm_infonce": loss_cm_infonce,
            "cm_triplet": loss_cm_triplet,
        }
