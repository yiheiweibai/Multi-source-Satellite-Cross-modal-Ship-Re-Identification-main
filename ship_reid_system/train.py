"""训练入口：python train.py [--config configs/xxx.yaml]

流程：加载配置 -> 生成/加载数据 -> 构建模型 -> 训练 + 周期验证 -> 保存 checkpoint。
数据开放前 dummy_mode=true 自动生成模拟数据，跑通全流程。
官方 labels.csv 接入：config 填 train_labels_csv（绝对路径），dummy_mode=false，
val_labels_csv 留空则跳过验证；身份数由数据集自动推断。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import (
    PKSampler,
    ShipReIDCSVDataset,
    ShipReIDDataset,
    build_transforms,
    ensure_dummy_data,
)
from models import build_model
from trainers import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ship ReID Training")
    p.add_argument("--config", type=str, default="", help="自定义 yaml 配置路径")
    p.add_argument("--resume", type=str, default="", help="checkpoint 路径")
    p.add_argument("--epochs", type=int, default=0, help="覆盖训练轮数")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_speckle(cfg):
    """SAR 斑点增强：仅训练集、且开关打开时启用。"""
    if not cfg.sar_speckle_aug:
        return None
    from data.transforms import SimulateSpeckle

    return SimulateSpeckle(noise_level=0.15, p=0.5)


def _is_csv_mode(cfg) -> bool:
    return bool(getattr(cfg, "train_labels_csv", ""))


def build_loaders(cfg):
    """构建 train/val DataLoader（PK 采样）。

    CSV 模式（train_labels_csv 非空）：
        - train 数据集来自官方 labels.csv（image_path 相对 csv 所在数据包根目录）
        - max_samples > 0 时截取前 N 个样本（冒烟/调试）
        - val_labels_csv 为空 -> val_loader=None（训练时不验证）
    JSON 模式：维持原逻辑（dummy 或自建标注）。
    """
    if cfg.dummy_mode:
        ensure_dummy_data(cfg)

    if _is_csv_mode(cfg):
        train_ds = ShipReIDCSVDataset(
            cfg.train_labels_csv,
            transform=build_transforms(cfg, is_train=True),
            sar_speckle=build_speckle(cfg),
            is_train=True,
            max_samples=getattr(cfg, "max_samples", 0),
        )
        if getattr(cfg, "val_labels_csv", ""):
            val_ds = ShipReIDCSVDataset(
                cfg.val_labels_csv,
                transform=build_transforms(cfg, is_train=False),
                sar_speckle=None,
                is_train=False,
                max_samples=getattr(cfg, "max_samples", 0),
            )
        else:
            val_ds = None
    else:
        train_ds = ShipReIDDataset(
            cfg.train_ann,
            transform=build_transforms(cfg, is_train=True),
            sar_speckle=build_speckle(cfg),
            is_train=True,
        )
        val_ds = ShipReIDDataset(
            cfg.val_ann,
            transform=build_transforms(cfg, is_train=False),
            sar_speckle=None,
            is_train=False,
        )

    # 采样器：pk_sampler=cross_modal 时使用模态均衡 CrossModalPKSampler（默认 pk）
    sampler_cls = CrossModalPKSampler if getattr(cfg, "pk_sampler", "pk") == "cross_modal" else PKSampler
    sampler = sampler_cls(
        train_ds.labels(),
        p=cfg.pk_p,
        k=cfg.pk_k,
        modalities=train_ds.modalities(),
        prefer_dual_modal=cfg.pk_prefer_dual_modal,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.pk_p * cfg.pk_k,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=64,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config or "config/default.yaml")
    if args.resume:
        cfg.resume = args.resume
    if args.epochs > 0:
        cfg.epochs = args.epochs

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dummy_mode={cfg.dummy_mode} | backbone={cfg.backbone}")

    train_loader, val_loader = build_loaders(cfg)

    # 训练身份数自动推断（覆盖 num_classes，保证分类头维度正确）
    if getattr(cfg, "auto_num_classes", True) and hasattr(train_loader.dataset, "num_classes"):
        inferred = int(train_loader.dataset.num_classes())
        if inferred != cfg.num_classes:
            print(f"auto_num_classes: {cfg.num_classes} -> {inferred}")
        cfg.num_classes = inferred

    model = build_model(cfg)
    trainer = Trainer(cfg, model, train_loader, val_loader, device)
    best_map = trainer.train(cfg.epochs)
    print(f"Training finished. Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
