"""模型组装：双分支骨干 + 模态不变特征头。

forward(x, modality) -> (ret_feat, logits)
- ret_feat: L2 归一化检索特征，用于 SupCon/三元组损失与检索
- logits: 身份分类输出，用于 CE 损失

多尺度特征提取（推理端增强）：
- extract_multiscale_feature: 全局 + 中心局部 + 水平条带特征融合

方向特异性投影（Phase 4）：
- extract_directional_feature: 按 query_type 选择 O2S / S2O 方向投影
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import BNNeckHead, ModalityAwareHead, DirectionalProjectionHead


class ShipReIDModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(cfg)
        emb_dim = getattr(cfg, "embedding_dim", 0)
        num_classes = getattr(cfg, "num_classes", 1000)
        # 实验开关 modality_projection：开启时使用模态感知头（O/S 独立投影）
        if getattr(cfg, "modality_projection", False):
            self.head = ModalityAwareHead(
                self.backbone.out_dim,
                num_classes,
                embedding_dim=emb_dim,
                modality_projection=True,
            )
        else:
            self.head = BNNeckHead(self.backbone.out_dim, num_classes, embedding_dim=emb_dim)
        # ArcFace 头（w_arcface > 0 时启用）
        if getattr(cfg, "w_arcface", 0.0) > 0:
            self.head.set_arcface(
                num_classes,
                scale=getattr(cfg, "arcface_scale", 30.0),
                margin=getattr(cfg, "arcface_margin", 0.5),
            )
        self.use_arcface = getattr(cfg, "w_arcface", 0.0) > 0

        # Phase 4：方向特异性非对称投影头（directional_projection=true 时启用）
        self.use_directional = getattr(cfg, "directional_projection", False)
        if self.use_directional:
            feat_dim = emb_dim if emb_dim > 0 else self.backbone.out_dim
            self.directional_head = DirectionalProjectionHead(feat_dim)

    def forward(
        self, x: torch.Tensor, modality: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """返回 (ret_feat, logits, arcface_logits)。arcface_logits 仅训练时需要 labels。"""
        feat = self.backbone(x, modality)
        ret_feat, logits = self.head(feat, modality)
        arcface_logits = None
        if self.use_arcface and labels is not None and self.training:
            arcface_logits = self.head.arcface_forward(feat, labels, modality)
        return ret_feat, logits, arcface_logits

    @torch.no_grad()
    def extract_feature(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        """推理特征提取（自动 eval 模式）。"""
        was_training = self.training
        self.eval()
        ret_feat, _, _ = self.forward(x, modality)
        if was_training:
            self.train()
        return ret_feat

    @torch.no_grad()
    def extract_directional_feature(
        self, x: torch.Tensor, modality: torch.Tensor, query_type: str
    ) -> torch.Tensor:
        """按 query_type 提取方向特异性特征（Phase 4）。

        Args:
            query_type: "O2S" / "S2O" / "O2O"

        未启用 directional_projection 时回退到标准特征。
        """
        if not self.use_directional:
            return self.extract_feature(x, modality)

        was_training = self.training
        self.eval()
        feat = self.backbone(x, modality)
        ret_feat, _ = self.head(feat, modality)
        dir_feat = self.directional_head.extract_directional(ret_feat, query_type)
        if was_training:
            self.train()
        return dir_feat

    @torch.no_grad()
    def extract_multiscale_feature(
        self, x: torch.Tensor, modality: torch.Tensor,
        global_weight: float = 0.5, center_weight: float = 0.3, stripe_weight: float = 0.2,
    ) -> torch.Tensor:
        """多尺度特征提取：全局 + 中心局部 + 水平条带特征融合。"""
        was_training = self.training
        self.eval()

        # 1. 全局特征（原始路径）
        feat_global, _, _ = self.forward(x, modality)

        # 2. 中心局部特征：先 crop 中央 50%，再 resize 回模型输入尺寸
        B, C, H, W = x.shape
        ch, cw = H // 4, W // 4  # 从 1/4 到 3/4，即中央 50%
        x_center = x[:, :, ch:H-ch, cw:W-cw]
        x_center = F.interpolate(x_center, size=(H, W), mode='bilinear', align_corners=False)
        feat_center, _, _ = self.forward(x_center, modality)

        # 3. 水平条带特征：上下两半分别 resize 后提取，取平均
        x_top = x[:, :, :H//2, :]
        x_bottom = x[:, :, H//2:, :]
        x_top = F.interpolate(x_top, size=(H, W), mode='bilinear', align_corners=False)
        x_bottom = F.interpolate(x_bottom, size=(H, W), mode='bilinear', align_corners=False)
        feat_top, _, _ = self.forward(x_top, modality)
        feat_bottom, _, _ = self.forward(x_bottom, modality)
        feat_stripe = (feat_top + feat_bottom) / 2

        # 加权融合（各特征已 L2 归一化，加权后再归一化）
        feat_fused = (
            global_weight * feat_global +
            center_weight * feat_center +
            stripe_weight * feat_stripe
        )
        feat_fused = F.normalize(feat_fused, dim=1)

        if was_training:
            self.train()
        return feat_fused


def build_model(cfg) -> ShipReIDModel:
    return ShipReIDModel(cfg)
