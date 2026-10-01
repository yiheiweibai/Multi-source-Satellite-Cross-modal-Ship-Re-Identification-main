"""训练/验证循环：难样本挖掘、checkpoint 保存/恢复、tensorboard 日志。

支持 --resume 恢复训练；每个 epoch 结束自动评估验证集 mAP/Recall@K。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses import ComposedLoss
from trainers.lr_scheduler import build_optimizer_and_scheduler
from utils.metrics import evaluate_retrieval
from utils.logger import get_logger


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = ComposedLoss(cfg).to(device)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(model, cfg)

        self.logger = get_logger("trainer")
        out_dir = Path(cfg.output_dir)
        self.log_dir = str(out_dir / "logs")
        self.ckpt_dir = out_dir / "checkpoints"
        self.writer = SummaryWriter(log_dir=self.log_dir) if cfg.tb_enabled else None
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.best_map = 0.0

        if getattr(cfg, "resume", "") and Path(cfg.resume).exists():
            self._load_checkpoint(cfg.resume)

    # ---------- 训练 ----------
    def train(self, epochs: int | None = None) -> float:
        epochs = epochs or getattr(self.cfg, "epochs", 60)
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            train_loss = 0.0
            train_n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, labels, modalities in pbar:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                modalities = modalities.to(self.device)

                ret_feat, logits = self.model(imgs, modalities)
                losses = self.criterion(ret_feat, logits, labels)
                self.optimizer.zero_grad()
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                train_loss += losses["loss"].item() * imgs.size(0)
                train_n += imgs.size(0)
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")
            avg_loss = train_loss / max(train_n, 1)

            # 验证
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)

            # 日志
            self.logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f} "
                f"mAP={val_metrics.get('mAP', 0):.4f} "
                f"R1={val_metrics.get('R1', 0):.4f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("train/loss", avg_loss, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            # checkpoint（无验证集时也保存 best，供推理直接使用）
            map_score = val_metrics.get("mAP", 0.0)
            self._save_checkpoint(epoch, is_best=((map_score > self.best_map) or (self.val_loader is None)))
            if map_score > self.best_map:
                self.best_map = map_score
            if self.scheduler is not None:
                self.scheduler.step()
        return self.best_map

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        feats, labels, mods = [], [], []
        for imgs, lab, mod in loader:
            imgs = imgs.to(self.device)
            mod = mod.to(self.device)
            feat = self.model.extract_feature(imgs, mod)
            feats.append(feat.cpu())
            labels.append(lab)
            mods.append(mod.cpu())
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        mods = torch.cat(mods)
        return evaluate_retrieval(feats, labels, mods, k_values=[1, 5, 10])

    # ---------- checkpoint ----------
    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_map": self.best_map,
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else None,
        }
        path = self.ckpt_dir / "last.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        self.logger.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state and state["optimizer_state"]:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state and state["scheduler_state"] and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.start_epoch = state.get("epoch", 0)
        self.best_map = state.get("best_map", 0.0)
        self.logger.info(f"Resumed from {path}, epoch={self.start_epoch}")
