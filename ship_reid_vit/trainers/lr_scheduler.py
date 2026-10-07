"""分层学习率、Warmup + Cosine 调度器。

分层 LR：骨干浅层用小 LR、深层与头用大 LR。
Warmup：前 warmup_epochs 线性升温到目标 LR，随后 Cosine 衰减。
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR


class WarmupCosineScheduler:
    """线性 Warmup + Cosine 退火。step() 按 epoch 调用。"""

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int, base_lrs: list[float]) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(1, total_epochs)
        self.base_lrs = base_lrs
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.last_epoch <= self.warmup_epochs:
                lr = base_lr * (self.last_epoch / self.warmup_epochs)
            else:
                progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            group["lr"] = lr

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = state.get("last_epoch", 0)


def build_optimizer_and_scheduler(
    model: nn.Module, cfg
) -> tuple[Optimizer, object]:
    backbone_lr_scale = getattr(cfg, "backbone_lr_scale", 0.1)
    lr = getattr(cfg, "lr", 3e-4)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    base_lrs = [lr * backbone_lr_scale, lr]
    param_groups = [
        {"params": backbone_params, "lr": base_lrs[0]},
        {"params": head_params, "lr": base_lrs[1]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=getattr(cfg, "weight_decay", 1e-4)
    )

    total_steps = getattr(cfg, "total_steps", 0)
    scheduler_type = getattr(cfg, "scheduler", "cosine")
    warmup_epochs = getattr(cfg, "warmup_epochs", 1)
    epochs = getattr(cfg, "epochs", 60)

    if scheduler_type == "multistep":
        milestones = [int(epochs * m) for m in (0.5, 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_type == "cosine" and total_steps > 0:
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=base_lrs,
            total_steps=total_steps,
            pct_start=max(0.05, warmup_epochs / epochs),
        )
    else:
        # 默认：Warmup + Cosine（按 epoch）
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, base_lrs)

    return optimizer, scheduler
