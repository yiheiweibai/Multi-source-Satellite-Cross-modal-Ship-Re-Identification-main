"""训练/验证循环：难样本挖掘、EMA、AMP、checkpoint 保存/恢复。"""
from __future__ import annotations

import copy
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


class EMA:
    """模型参数指数滑动平均。推理时使用 EMA 权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def apply(self, model: nn.Module) -> None:
        """将 EMA 权重复制到模型（推理前调用）。"""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = ComposedLoss(cfg).to(device)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(model, cfg)

        # Phase 4：方向特异性投影的辅助损失（directional_projection=true 时启用）。
        # 关键：投影头必须收到梯度，否则推理端 --directional 只是随机投影。
        # O2S 方向损失只流经 proj_o2s、S2O 只流经 proj_s2o（方向特定空间）。
        self.use_directional = bool(
            getattr(cfg, "directional_projection", False)
        ) and hasattr(model, "directional_head")
        if self.use_directional:
            from losses import CrossModalInfoNCE
            temp = float(getattr(cfg, "cm_temperature", 0.07))
            self.dir_infonce_o2s = CrossModalInfoNCE(temperature=temp, w_os=1.0, w_so=0.0)
            self.dir_infonce_s2o = CrossModalInfoNCE(temperature=temp, w_os=0.0, w_so=1.0)
            self.w_directional = float(getattr(cfg, "w_directional", 0.3))

        # EMA
        self.use_ema = getattr(cfg, "use_ema", True)
        self.ema = EMA(model, decay=getattr(cfg, "ema_decay", 0.999)) if self.use_ema else None

        # AMP
        self.use_amp = getattr(cfg, "amp", True) and device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.logger = get_logger("trainer")
        out_dir = Path(cfg.output_dir)
        self.log_dir = str(out_dir / "logs")
        self.ckpt_dir = out_dir / "checkpoints"
        self.writer = SummaryWriter(log_dir=self.log_dir) if cfg.tb_enabled else None
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(getattr(cfg, "save_every", 0))   # >0 时每 N 个 epoch 另存一份
        self.start_epoch = 0
        self.best_map = 0.0

        if getattr(cfg, "resume", "") and Path(cfg.resume).exists():
            self._load_checkpoint(cfg.resume)

    # ---------- 训练 ----------
    def train(self, epochs: int | None = None) -> float:
        epochs = epochs or getattr(self.cfg, "epochs", 60)
        # Phase 2：跨模态硬负样本挖掘间隔（0=禁用）
        hn_interval = int(getattr(self.cfg, "hard_neg_mining_interval", 0))
        hn_topk = int(getattr(self.cfg, "hard_neg_topk", 5))
        for epoch in range(self.start_epoch, epochs):
            # Phase 2：周期性执行硬负样本挖掘
            if hn_interval > 0 and epoch > 0 and epoch % hn_interval == 0:
                hard_ids = self._mine_cross_modal_hard_negatives(topk=hn_topk)
                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_hard_negatives"):
                    sampler.set_hard_negatives(hard_ids)
                    self.logger.info(
                        f"[Phase2] Epoch {epoch + 1}: 挖掘到 {len(hard_ids)} 个硬负样本身份，"
                        f"已注入 PK 采样器"
                    )

            self.model.train()
            train_loss = 0.0
            train_n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, labels, modalities in pbar:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                modalities = modalities.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    ret_feat, logits, arcface_logits = self.model(imgs, modalities, labels=labels)
                    losses = self.criterion(
                        ret_feat, logits, labels,
                        modalities=modalities,
                        arcface_logits=arcface_logits,
                    )

                total_loss = losses["loss"]
                if self.use_directional:
                    # Phase 4：方向特异性辅助损失（autocast 外 fp32 计算，数值稳定）
                    dir_loss = self._directional_aux_loss(ret_feat, labels, modalities)
                    total_loss = total_loss + self.w_directional * dir_loss
                    losses["dir"] = dir_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.ema is not None:
                    self.ema.update(self.model)

                train_loss += total_loss.item() * imgs.size(0)
                train_n += imgs.size(0)
                postfix = {"loss": f"{total_loss.item():.4f}"}
                if self.use_directional:
                    postfix["dir"] = f"{dir_loss.item():.3f}"
                pbar.set_postfix(**postfix)
            avg_loss = train_loss / max(train_n, 1)

            # 验证（用 EMA 权重）
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)

            self.logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f} "
                f"mAP={val_metrics.get('mAP', 0):.4f} "
                f"R1={val_metrics.get('R1', 0):.4f}"
            )
            if self.writer is not None:
                self.writer.add_scalar("train/loss", avg_loss, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

            map_score = val_metrics.get("mAP", 0.0)
            periodic = self.save_every > 0 and (epoch + 1) % self.save_every == 0
            self._save_checkpoint(
                epoch,
                is_best=((map_score > self.best_map) or (self.val_loader is None)),
                periodic=periodic,
            )
            if map_score > self.best_map:
                self.best_map = map_score
            if self.scheduler is not None:
                self.scheduler.step()
        return self.best_map

    @torch.no_grad()
    def _mine_cross_modal_hard_negatives(self, topk: int = 5) -> set:
        """跨模态硬负样本挖掘：用当前模型计算训练集跨模态相似度，为每个身份找出 top-k 最难区分的跨模态负样本身份。

        返回：硬负样本身份集合（并集）。
        """
        self.model.eval()
        feats, labels, mods = [], [], []
        for imgs, lab, mod in self.train_loader:
            imgs = imgs.to(self.device)
            mod = mod.to(self.device)
            feat = self.model.extract_feature(imgs, mod)
            feats.append(feat.cpu())
            labels.append(lab)
            mods.append(mod.cpu())
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        mods = torch.cat(mods)

        # 按模态拆分
        o_mask = mods == 0
        s_mask = mods == 1
        o_feat, o_lab = feats[o_mask], labels[o_mask]
        s_feat, s_lab = feats[s_mask], labels[s_mask]
        if o_feat.size(0) == 0 or s_feat.size(0) == 0:
            return set()

        # 计算跨模态相似度矩阵（光学 x SAR）
        o_n = torch.nn.functional.normalize(o_feat, dim=1)
        s_n = torch.nn.functional.normalize(s_feat, dim=1)
        sim = o_n @ s_n.t()  # (No, Ns)

        hard_ids = set()
        # O2S：为每个光学身份找最难区分的 SAR 身份（同身份为正，其余为负）
        for i in range(o_n.size(0)):
            pos_mask = s_lab == o_lab[i]
            neg_sim = sim[i].clone()
            neg_sim[pos_mask] = -1.0  # 排除正样本
            _, top_neg_idx = torch.topk(neg_sim, min(topk, s_n.size(0)))
            hard_ids.update(s_lab[top_neg_idx].tolist())
        # S2O：为每个 SAR 身份找最难区分的光学身份
        for j in range(s_n.size(0)):
            pos_mask = o_lab == s_lab[j]
            neg_sim = sim[:, j].clone()
            neg_sim[pos_mask] = -1.0
            _, top_neg_idx = torch.topk(neg_sim, min(topk, o_n.size(0)))
            hard_ids.update(o_lab[top_neg_idx].tolist())

        return hard_ids

    def _directional_aux_loss(
        self, ret_feat: torch.Tensor, labels: torch.Tensor, modalities: torch.Tensor
    ) -> torch.Tensor:
        """Phase 4：方向特异性 InfoNCE 辅助损失。

        O2S 方向（光学 anchor → SAR）：在 proj_o2s 空间内做单向跨模态 InfoNCE；
        S2O 方向（SAR anchor → 光学）：在 proj_s2o 空间内做单向跨模态 InfoNCE。
        两方向损失分别只回传各自的投影头，保证投影头学到方向特定的判别空间。
        """
        feat = ret_feat.float()
        f_o2s, f_s2o = self.model.directional_head(feat)
        l_o2s = self.dir_infonce_o2s(f_o2s, labels, modalities)
        l_s2o = self.dir_infonce_s2o(f_s2o, labels, modalities)
        return 0.5 * (l_o2s + l_s2o)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        # 评估时使用 EMA 权重
        if self.ema is not None:
            backup = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.ema.apply(self.model)
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
        # 恢复训练权重
        if self.ema is not None:
            self.model.load_state_dict(backup)
        return evaluate_retrieval(feats, labels, mods, k_values=[1, 5, 10])

    # ---------- checkpoint ----------
    def _save_checkpoint(self, epoch: int, is_best: bool, periodic: bool = False) -> None:
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema else None,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
            "best_map": self.best_map,
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else None,
        }
        path = self.ckpt_dir / "last.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
        if periodic:
            periodic_path = self.ckpt_dir / f"epoch_{epoch + 1:03d}.pth"
            torch.save(state, periodic_path)
            self.logger.info(f"Saved periodic checkpoint: {periodic_path}")
        self.logger.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if self.ema is not None and state.get("ema_state"):
            self.ema.load_state_dict(state["ema_state"])
        if "optimizer_state" in state and state["optimizer_state"]:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state and state["scheduler_state"] and self.scheduler:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self.use_amp and state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])
        self.start_epoch = state.get("epoch", 0)
        self.best_map = state.get("best_map", 0.0)
        self.logger.info(f"Resumed from {path}, epoch={self.start_epoch}")
