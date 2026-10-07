"""数据增强与 SAR 斑点滤波。

模态独立归一化：optical 与 SAR 分别使用各自 mean/std（由 compute_stats.py 生成）。
SAR 专属增强：禁用 ColorJitter，改用强度抖动 + 弹性形变。

提供统一接口 ShipTransform(img, modality) -> tensor，dataset 直接调用。
"""
from __future__ import annotations

import torch
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimulateSpeckle:
    """模拟 SAR 乘性相干斑噪声（强度域 Gamma 噪声近似）。"""

    def __init__(self, noise_level: float = 0.15, p: float = 0.5) -> None:
        self.noise_level = noise_level
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        l_equiv = max(1.0, 1.0 / (self.noise_level ** 2))
        noise = torch.distributions.Gamma(
            torch.tensor(float(l_equiv)), torch.tensor(float(l_equiv))
        ).sample(img.shape)
        return img * noise.to(img.device)


class SARIntensityJitter:
    """SAR 强度抖动：仅调整亮度/对比度，不动色相饱和度。"""

    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, p: float = 0.5) -> None:
        self.t = T.ColorJitter(brightness=brightness, contrast=contrast)
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        return self.t(img)


class ShipTransform:
    """统一变换：__call__(img, modality) -> tensor。

    modality: 0=optical, 1=sar。训练时按模态应用不同光度增强与归一化。
    """

    def __init__(self, cfg, is_train: bool = True) -> None:
        self.is_train = is_train
        size = getattr(cfg, "image_size", 256)
        if getattr(cfg, "backbone", "resnet50").startswith("vit"):
            size = 224
        self.resize = (size, size)

        opt_mean = getattr(cfg, "optical_mean", IMAGENET_MEAN)
        opt_std = getattr(cfg, "optical_std", IMAGENET_STD)
        sar_mean = getattr(cfg, "sar_mean", IMAGENET_MEAN)
        sar_std = getattr(cfg, "sar_std", IMAGENET_STD)
        self.opt_norm = T.Normalize(mean=opt_mean, std=opt_std)
        self.sar_norm = T.Normalize(mean=sar_mean, std=sar_std)

        if is_train:
            self.common = T.Compose([
                T.RandomResizedCrop(self.resize, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ])
            self.optical_aug = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
                T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
            ])
            self.sar_aug = T.Compose([
                SARIntensityJitter(brightness=0.3, contrast=0.3, p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
                T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
            ])
        else:
            self.common = T.Compose([T.Resize(self.resize)])
            self.optical_aug = T.ToTensor()
            self.sar_aug = T.ToTensor()

    def __call__(self, img, modality: int):
        img = self.common(img)
        if modality == 1:  # sar
            img = self.sar_aug(img)
            img = self.sar_norm(img)
        else:  # optical
            img = self.optical_aug(img)
            img = self.opt_norm(img)
        return img


def build_transforms(cfg, is_train: bool = True):
    """返回 ShipTransform 实例（callable: img, modality -> tensor）。"""
    return ShipTransform(cfg, is_train=is_train)
