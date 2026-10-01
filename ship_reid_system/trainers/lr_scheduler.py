"""分层学习率与优化器/调度器。

分层 LR：骨干浅层用小 LR、深层与头用大 LR（对应技术方案分层微调策略）。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def build_optimizer_and_scheduler(
    model: nn.Module, cfg
) -> tuple[Optimizer, object]:
    backbone_lr_scale = getattr(cfg, "backbone_lr_scale", 0.1)
    lr = getattr(cfg, "lr", 3e-4)

    # 分层：backbone 用 lr*scale，其余（head 等）用 lr
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr * backbone_lr_scale},
        {"params": head_params, "lr": lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=getattr(cfg, "weight_decay", 1e-4)
    )

    total_steps = getattr(cfg, "total_steps", 0)
    scheduler_type = getattr(cfg, "scheduler", "cosine")
    if scheduler_type == "multistep":
        milestones = [int(getattr(cfg, "epochs", 60) * m) for m in (0.5, 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_type == "cosine" and total_steps > 0:
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.1,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(getattr(cfg, "epochs", 60), 1))

    return optimizer, scheduler
