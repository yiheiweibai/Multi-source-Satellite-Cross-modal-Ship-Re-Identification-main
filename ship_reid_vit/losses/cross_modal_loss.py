"""跨模态特征学习损失：Cross-Modal InfoNCE 与 Cross-Modal Hard Triplet。

输入约定：
- features: 已做 L2 归一化的检索特征 (B, D)
- labels: 身份标签 (B,)
- modalities: 模态标签 (B,)，0=光学(Optical) 1=SAR

设计依据（优化路线1.md 第五/六节）：
- 本赛题 90% 以上身份仅 Optical 1 张 + SAR 1 张，天然跨模态 positive pair；
- CrossModalInfoNCE 构造 O/S 对齐矩阵 M = O'·S'^T，对角线拉近、非对角线推远；
- CrossModalHardTriplet 只跨模态构造正负对，贴近 O2S / S2O 官方测试环境。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import cross_modal_hard_triplet as _cross_modal_hard_triplet


def _split_dual_modal(features, labels, modalities):
    """按模态拆分样本，并过滤为仅含"批内双模态身份"的样本。

    返回 (O_feat, O_lab, S_feat, S_lab)；任一模态为空或无双模态身份时返回 None。
    单模态身份样本不参与跨模态损失（不影响梯度）。
    """
    o_mask = modalities == 0
    s_mask = modalities == 1
    o_feat, o_lab = features[o_mask], labels[o_mask]
    s_feat, s_lab = features[s_mask], labels[s_mask]
    if o_feat.size(0) == 0 or s_feat.size(0) == 0:
        return None
    dual = set(o_lab.tolist()) & set(s_lab.tolist())
    if not dual:
        return None
    dual_t = torch.tensor(sorted(dual), device=features.device, dtype=labels.dtype)
    o_keep = torch.isin(o_lab, dual_t)
    s_keep = torch.isin(s_lab, dual_t)
    return o_feat[o_keep], o_lab[o_keep], s_feat[s_keep], s_lab[s_keep]


class CrossModalInfoNCE(nn.Module):
    """跨模态 InfoNCE：批内按模态拆 O/S 两组，构造 M = O'·S'^T。

    双向 CE：CE(M / temperature, labels) + CE(M.T / temperature, labels) 平均。
    positive 为同身份跨模态对、negative 为不同身份；仅计算双模态身份，
    支持批内同一身份多个样本（多 positive 按行归一化权重）。
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, modalities: torch.Tensor
    ) -> torch.Tensor:
        split = _split_dual_modal(features, labels, modalities)
        if split is None:
            return features.new_tensor(0.0)
        o_feat, o_lab, s_feat, s_lab = split

        # 防御性再归一化（调用方通常已 L2 归一化）
        o_n = F.normalize(o_feat, dim=1)
        s_n = F.normalize(s_feat, dim=1)
        sim = o_n @ s_n.t() / self.temperature  # (No, Ns)

        log_os = F.log_softmax(sim, dim=1)
        log_so = F.log_softmax(sim, dim=0)

        # 多 positive 目标权重矩阵：行/列内同身份样本均分
        pos = (o_lab.unsqueeze(1) == s_lab.unsqueeze(0)).float()  # (No, Ns)
        target_os = pos / pos.sum(dim=1, keepdim=True).clamp(min=1.0)
        target_so = pos.t() / pos.t().sum(dim=1, keepdim=True).clamp(min=1.0)

        loss_os = -(target_os * log_os).sum(dim=1).mean()
        loss_so = -(target_so * log_so).sum(dim=1).mean()
        return 0.5 * (loss_os + loss_so)


class CrossModalHardTriplet(nn.Module):
    """跨模态 hard triplet（包装 triplet_loss.cross_modal_hard_triplet）。

    双向难样本挖掘：O→S（anchor=光学，正=同身份 SAR 最近，负=不同身份 SAR 最近）
    与 S→O 方向平均；某身份缺另一模态样本时该 anchor 自动跳过。
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, modalities: torch.Tensor
    ) -> torch.Tensor:
        return _cross_modal_hard_triplet(features, labels, modalities, margin=self.margin)
