"""跨模态监督对比损失（SupCon）。

与经典 SupCon 区别：正样本对按"身份"定义，同时鼓励
- 同身份不同模态样本拉近（跨模态对齐）
- 同模态不同身份样本推开（细粒度区分）
- temperature 可调；训练时配合 head 输出的 L2 归一化特征
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """features: (N, D) L2 归一化特征；labels: (N,) 身份标签。"""
        device = features.device
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)

        if mask is None:
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)
        # 去掉自身
        eye = torch.eye(batch_size, device=device).bool()
        mask = mask.masked_fill(eye, 0.0)

        sim = torch.matmul(features, features.t()) / self.temperature  # (N, N)

        # 数值稳定性
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # 对数分母：除自身外的所有样本（经典 SupCon 分母排除自身；减去 e^0=1 的自身项）
        exp_logits = torch.exp(logits)
        denom = (exp_logits.sum(dim=1, keepdim=True) - 1.0).clamp(min=1e-12)
        log_prob = logits - torch.log(denom)

        # 分子：正样本对数概率均值
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob_pos
        # 只对有正样本的样本计算
        has_pos = mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        loss = loss[has_pos].mean()
        loss = loss * self.temperature / self.base_temperature
        return loss
