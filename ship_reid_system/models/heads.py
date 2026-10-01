"""特征头：BNNeck 风格模态不变特征头。

- 检索特征：BN 前的原始特征做 L2 归一化（保留判别性，避免 BN 抹平模态差异）
- 分类 logits：BN 后特征接全连接（BNNeck 思想，分类与度量解耦）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeckHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.bn.bias.requires_grad_(False)  # BNNeck 标准设置
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_bn = self.bn(feat)
        logits = self.classifier(feat_bn)
        # BN 前特征归一化用于检索
        ret_feat = F.normalize(feat, dim=1)
        return ret_feat, logits
