"""CycleGAN 数据集：unpaired optical(A) / sar(B) 两域。

用法：
    dataset = UnpairedDataset(root="cyclegan_data", phase="train", load_size=286, crop_size=256)

目录结构：
    cyclegan_data/
    ├── trainA/  # optical 图像
    ├── trainB/  # sar 图像
    ├── testA/
    └── testB/
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


class UnpairedDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", load_size: int = 286,
                 crop_size: int = 256, no_flip: bool = False, serial: bool = False):
        self.root = Path(root)
        self.serial = serial
        dir_a = self.root / f"{phase}A"
        dir_b = self.root / f"{phase}B"
        self.A_paths = list_images(dir_a)
        self.B_paths = list_images(dir_b)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert self.A_size > 0 and self.B_size > 0, f"空目录: {dir_a}({self.A_size}) {dir_b}({self.B_size})"

        # 图像归一化：SAR 虽为灰度复制三通道，但值域 [0,1] 归一化后与 optical 分布接近
        transform_list = [transforms.Resize(load_size, transforms.InterpolationMode.BICUBIC)]
        if phase == "train":
            transform_list += [transforms.RandomCrop(crop_size)]
            if not no_flip:
                transform_list += [transforms.RandomHorizontalFlip()]
                transform_list += [transforms.RandomVerticalFlip()]
        else:
            transform_list += [transforms.CenterCrop(crop_size)]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_path = self.A_paths[index % self.A_size]
        if self.serial:
            b_path = self.B_paths[index % self.B_size]
        else:
            b_path = self.B_paths[random.randint(0, self.B_size - 1)]
        a_img = self._load(a_path)
        b_img = self._load(b_path)
        return {"A": a_img, "B": b_img, "A_paths": str(a_path), "B_paths": str(b_path)}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def _load(self, path: Path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)
