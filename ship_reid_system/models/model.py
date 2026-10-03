"""模型组装：双分支骨干 + 模态不变特征头。

forward(x, modality) -> (ret_feat, logits)
- ret_feat: L2 归一化检索特征，用于 SupCon/三元组损失与检索
- logits: 身份分类输出，用于 CE 损失
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone
from .heads import BNNeckHead, ModalityAwareHead


class ShipReIDModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)
        emb_dim = getattr(cfg, "embedding_dim", 0)
        num_classes = getattr(cfg, "num_classes", 1000)
        # 实验开关 modality_projection：开启时使用模态感知头（O/S 独立投影）
        if getattr(cfg, "modality_projection", False):
            self.head = ModalityAwareHead(
                self.backbone.out_dim,
                num_classes,
                embedding_dim=emb_dim,
                modality_projection=True,
            )
        else:
            self.head = BNNeckHead(self.backbone.out_dim, num_classes, embedding_dim=emb_dim)
        # ArcFace 头（w_arcface > 0 时启用）
        if getattr(cfg, "w_arcface", 0.0) > 0:
            self.head.set_arcface(
                num_classes,
                scale=getattr(cfg, "arcface_scale", 30.0),
                margin=getattr(cfg, "arcface_margin", 0.5),
            )
        self.use_arcface = getattr(cfg, "w_arcface", 0.0) > 0

    def forward(
        self, x: torch.Tensor, modality: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """返回 (ret_feat, logits, arcface_logits)。arcface_logits 仅训练时需要 labels。"""
        feat = self.backbone(x, modality)
        ret_feat, logits = self.head(feat, modality)
        arcface_logits = None
        if self.use_arcface and labels is not None and self.training:
            arcface_logits = self.head.arcface_forward(feat, labels, modality)
        return ret_feat, logits, arcface_logits

    @torch.no_grad()
    def extract_feature(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        """推理特征提取（自动 eval 模式）。"""
        was_training = self.training
        self.eval()
        ret_feat, _, _ = self.forward(x, modality)
        if was_training:
            self.train()
        return ret_feat


def build_model(cfg) -> ShipReIDModel:
    return ShipReIDModel(cfg)
