"""数据增强与 SAR 斑点滤波占位。

设计原则：不破坏船体几何结构（翻转/旋转/缩放/强度抖动/遮挡模拟），
对 SAR 分支额外施加斑点噪声模拟，增强对成像条件的鲁棒性。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torchvision import transforms as T

# 占位：后续可替换为真实光学统计；数据开放后建议按模态分别统计
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SpeckleFilterPlaceholder:
    """SAR 斑点滤波占位模块。

    当前为恒等变换，仅保留接口。数据开放后可替换为：
    - Lee / Kuan / Frost 自适应滤波
    - BM3D 或 SAR-BM3D
    - 基于深度学习的去斑网络（如 Speckle2Void）

    使用方式：在 build_transforms 中启用，或单独对 SAR 图调用。
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img


class SimulateSpeckle:
    """模拟 SAR 乘性相干斑噪声（强度域 Gamma 噪声近似）。

    用于训练增强，使模型对真实斑点干扰鲁棒。
    """

    def __init__(self, noise_level: float = 0.15, p: float = 0.5) -> None:
        self.noise_level = noise_level
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        # 乘性噪声：L 视数对应的 Gamma 分布，noise_level 越小视数越大
        l_equiv = max(1.0, 1.0 / (self.noise_level ** 2))
        noise = torch.distributions.Gamma(
            torch.tensor(float(l_equiv)), torch.tensor(float(l_equiv))
        ).sample(img.shape)
        return img * noise.to(img.device)


def build_transforms(cfg, is_train: bool = True) -> Callable:
    """构建训练/测试变换。image_size 按骨干自动调整（ViT 用 224）。"""
    size = getattr(cfg, "image_size", 256)
    if getattr(cfg, "backbone", "resnet50").startswith("vit"):
        size = 224
    resize = (size, size)
    if is_train:
        return T.Compose(
            [
                # 随机裁剪缩放：增加尺度变化，配合 PK 重复索引生成更多样虚拟视图
                T.RandomResizedCrop(resize, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                # 光照/强度抖动：对光学近似光照变化，对 SAR 近似辐射变化
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
                # 随机擦除模拟遮挡、并靠截断、云雾遮挡（作用于 Tensor）
                T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
