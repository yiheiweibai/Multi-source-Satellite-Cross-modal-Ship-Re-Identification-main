"""模型组装：双分支骨干 + 模态不变特征头。

forward(x, modality) -> (ret_feat, logits)
- ret_feat: L2 归一化检索特征，用于 SupCon/三元组损失与检索
- logits: 身份分类输出，用于 CE 损失
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads import BNNeckHead


class ShipReIDModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.head = BNNeckHead(self.backbone.out_dim, getattr(cfg, "num_classes", 1000))

    def forward(
        self, x: torch.Tensor, modality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x, modality)
        ret_feat, logits = self.head(feat)
        return ret_feat, logits

    @torch.no_grad()
    def extract_feature(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        """推理特征提取（自动 eval 模式）。"""
        was_training = self.training
        self.eval()
        ret_feat, _ = self.forward(x, modality)
        if was_training:
            self.train()
        return ret_feat


def build_model(cfg) -> ShipReIDModel:
    return ShipReIDModel(cfg)
