"""特征头：BNNeck 风格模态不变特征头。

- 检索特征：BN 前的原始特征做 L2 归一化（保留判别性，避免 BN 抹平模态差异）
- 分类 logits：BN 后特征接全连接（BNNeck 思想，分类与度量解耦）
- 可选投影头：将骨干特征投影到 embedding_dim（若 > 0）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeckHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, embedding_dim: int = 0) -> None:
        super().__init__()
        # 可选投影到 embedding_dim
        if embedding_dim and embedding_dim > 0 and embedding_dim != in_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            feat_dim = embedding_dim
        else:
            self.projection = None
            feat_dim = in_dim

        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)  # BNNeck 标准设置
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        # ArcFace 分类头（权重与 logits 分类头共享，但 forward 时加角度 margin）
        self.arcface_classifier = None

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projection is not None:
            feat = self.projection(feat)
        feat_bn = self.bn(feat)
        logits = self.classifier(feat_bn)
        # BN 前特征归一化用于检索
        ret_feat = F.normalize(feat, dim=1)
        return ret_feat, logits

    def set_arcface(self, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        """启用 ArcFace 头（与分类头共享权重结构，独立参数）。"""
        feat_dim = self.classifier.in_features
        self.arcface_classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.arcface_classifier.weight)
        self.arcface_scale = scale
        self.arcface_margin = margin

    def arcface_forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ArcFace logits：对正确类别角度加 margin，再 scale。"""
        import math
        feat_dim = self.classifier.in_features
        if self.projection is not None:
            feat = self.projection(feat)
        feat_bn = self.bn(feat)
        w = self.arcface_classifier.weight  # (C, D)
        # 余弦相似度
        cos = F.linear(F.normalize(feat_bn, dim=1), F.normalize(w, dim=1))  # (B, C)
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        # 对正确类别加 margin
        one_hot = F.one_hot(labels, num_classes=w.size(0)).float()
        theta = theta + self.arcface_margin * one_hot
        logits = self.arcface_scale * torch.cos(theta)
        return logits
