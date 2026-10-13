"""数据增强与 SAR 斑点滤波。

模态独立归一化：optical 与 SAR 分别使用各自 mean/std（由 compute_stats.py 生成）。
SAR 专属增强：禁用 ColorJitter，改用强度抖动 + 弹性形变 + 散射点感知增强。

提供统一接口 ShipTransform(img, modality) -> tensor，dataset 直接调用。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
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


class SARElasticDeform:
    """SAR 弹性形变：模拟 SAR 成像几何畸变。

    对图像应用轻微的弹性形变，保持船体整体结构但改变局部几何关系。
    使用随机位移场 + 双线性插值实现。
    """

    def __init__(self, alpha: float = 10.0, sigma: float = 3.0, p: float = 0.3) -> None:
        """
        Args:
            alpha: 形变强度（像素级）
            sigma: 高斯平滑核大小（控制形变平滑度）
            p: 应用概率
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        C, H, W = img.shape
        # 生成随机位移场
        dx = torch.randn(1, H, W, device=img.device) * self.alpha
        dy = torch.randn(1, H, W, device=img.device) * self.alpha
        # 高斯平滑使形变更自然
        kernel_size = int(self.sigma * 4) | 1  # 确保奇数
        dx = T.functional.gaussian_blur(dx, kernel_size, self.sigma)
        dy = T.functional.gaussian_blur(dy, kernel_size, self.sigma)
        # 构建采样网格
        yy, xx = torch.meshgrid(
            torch.arange(H, device=img.device, dtype=torch.float32),
            torch.arange(W, device=img.device, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = (xx + dx.squeeze(0)) / (W - 1) * 2 - 1
        grid_y = (yy + dy.squeeze(0)) / (H - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        # 双线性采样
        return F.grid_sample(
            img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(0)


class SARScatterAug:
    """SAR 散射点感知增强：针对 SAR 图像的稀疏散射点特性设计。

    组合两种轻量级增强（避免弹性形变的显存开销）：
    1. 随机局部对比度调整（模拟不同海况下的散射强度变化）
    2. 随机噪声注入（模拟 SAR 斑点噪声的不同水平）

    关键约束：保持"纯黑背景 + 稀疏散射点"的结构不被破坏。
    """

    def __init__(
        self,
        contrast_range: float = 0.4,
        noise_level_range: tuple = (0.05, 0.20),
        deform_alpha: float = 8.0,  # 保留参数兼容，但不使用（避免 OOM）
        p: float = 0.5,
    ) -> None:
        self.contrast_range = contrast_range
        self.noise_level_range = noise_level_range
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        C, H, W = img.shape

        # 1. 随机局部对比度调整：对高亮区域（散射点）和暗区分别处理
        # 找到图像中的高亮区域（散射点）
        threshold = img.quantile(0.95)
        bright_mask = img > threshold
        # 对高亮区域随机增强/减弱
        gain = 1.0 + (torch.rand(1).item() - 0.5) * self.contrast_range
        img = img.clone()
        img[bright_mask] = img[bright_mask] * gain
        # 对暗区随机微调
        dark_gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
        img[~bright_mask] = img[~bright_mask] * dark_gain

        # 2. 随机噪声注入（乘性噪声，模拟不同水平的斑点）
        noise_level = (
            self.noise_level_range[0] +
            torch.rand(1).item() * (self.noise_level_range[1] - self.noise_level_range[0])
        )
        l_equiv = max(1.0, 1.0 / (noise_level ** 2))
        noise = torch.distributions.Gamma(
            torch.tensor(float(l_equiv)), torch.tensor(float(l_equiv))
        ).sample(img.shape).to(img.device)
        img = img * noise

        # 确保像素值在合理范围内
        img = img.clamp(0, 1)
        return img


class ShipTransform:
    """统一变换：__call__(img, modality) -> tensor。

    modality: 0=optical, 1=sar。训练时按模态应用不同光度增强与归一化。
    """

    def __init__(self, cfg, is_train: bool = True) -> None:
        self.is_train = is_train
        size = getattr(cfg, "image_size", 256)
        if getattr(cfg, "backbone", "resnet50").startswith("vit"):
            size = getattr(cfg, "image_size", 224)
        self.resize = (size, size)

        opt_mean = getattr(cfg, "optical_mean", IMAGENET_MEAN)
        opt_std = getattr(cfg, "optical_std", IMAGENET_STD)
        sar_mean = getattr(cfg, "sar_mean", IMAGENET_MEAN)
        sar_std = getattr(cfg, "sar_std", IMAGENET_STD)
        self.opt_norm = T.Normalize(mean=opt_mean, std=opt_std)
        self.sar_norm = T.Normalize(mean=sar_mean, std=sar_std)

        # SAR 散射点感知增强开关（训练端）
        sar_scatter_aug = getattr(cfg, "sar_scatter_aug", False)

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
            sar_aug_list = [
                SARIntensityJitter(brightness=0.3, contrast=0.3, p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
            ]
            # SAR 散射点感知增强在 ToTensor 之后应用（需要 tensor 输入）
            if sar_scatter_aug:
                sar_aug_list.append(SARScatterAug(
                    contrast_range=0.4,
                    noise_level_range=(0.05, 0.20),
                    deform_alpha=8.0,
                    p=0.5,
                ))
            sar_aug_list.append(T.RandomErasing(p=0.4, scale=(0.02, 0.2)))
            self.sar_aug = T.Compose(sar_aug_list)
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
