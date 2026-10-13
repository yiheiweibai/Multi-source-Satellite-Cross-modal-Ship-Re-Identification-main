"""特征头：BNNeck 风格模态不变特征头。

- 检索特征：BN 前的原始特征做 L2 归一化（保留判别性，避免 BN 抹平模态差异）
- 分类 logits：BN 后特征接全连接（BNNeck 思想，分类与度量解耦）
- 可选投影头：将骨干特征投影到 embedding_dim（若 > 0）
- ModalityAwareHead：光学/SAR 独立 projection MLP 后进入共享 BNNeckHead（实验开关）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeckHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, embedding_dim: int = 0) -> None:
        super().__init__()
        # 可选投影到 embedding_dim
        if embedding_dim and embedding_dim > 0 and embedding_dim != in_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            feat_dim = embedding_dim
        else:
            self.projection = None
            feat_dim = in_dim

        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)  # BNNeck 标准设置
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        # ArcFace 分类头（权重与 logits 分类头共享，但 forward 时加角度 margin）
        self.arcface_classifier = None

    def forward(
        self, feat: torch.Tensor, modality: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # modality 仅用于统一接口（BNNeckHead 不区分模态），保持旧调用兼容
        if self.projection is not None:
            feat = self.projection(feat)
        feat_bn = self.bn(feat)
        logits = self.classifier(feat_bn)
        # BN 前特征归一化用于检索
        ret_feat = F.normalize(feat, dim=1)
        return ret_feat, logits

    def set_arcface(self, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        """启用 ArcFace 头（与分类头共享权重结构，独立参数）。"""
        feat_dim = self.classifier.in_features
        self.arcface_classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.arcface_classifier.weight)
        self.arcface_scale = scale
        self.arcface_margin = margin

    def arcface_forward(
        self, feat: torch.Tensor, labels: torch.Tensor, modality: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ArcFace logits：对正确类别角度加 margin，再 scale。"""
        import math
        feat_dim = self.classifier.in_features
        if self.projection is not None:
            feat = self.projection(feat)
        feat_bn = self.bn(feat)
        w = self.arcface_classifier.weight  # (C, D)
        # 余弦相似度
        cos = F.linear(F.normalize(feat_bn, dim=1), F.normalize(w, dim=1))  # (B, C)
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        # 对正确类别加 margin
        one_hot = F.one_hot(labels, num_classes=w.size(0)).float()
        theta = theta + self.arcface_margin * one_hot
        logits = self.arcface_scale * torch.cos(theta)
        return logits


class ModalityAwareHead(nn.Module):
    """模态感知头：光学/SAR 各自独立 projection MLP 后进入共享 BNNeckHead。

    - modality_projection=False（默认）：直接转发给内部 BNNeckHead，行为完全一致；
    - modality_projection=True：按 modality（0=光学 1=SAR）选择独立 projection
      MLP（Linear+BN+ReLU），输出再进入共享 BNNeckHead（BN+分类头）。
    检索特征取自模态投影后的原始特征做 L2 归一化（BNNeck 思想）。
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        embedding_dim: int = 0,
        modality_projection: bool = False,
    ) -> None:
        super().__init__()
        self.shared_head = BNNeckHead(in_dim, num_classes, embedding_dim=embedding_dim)
        self.modality_projection = modality_projection
        if modality_projection:
            self.opt_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.ReLU(inplace=True),
            )
            self.sar_proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.ReLU(inplace=True),
            )

    def _project(self, feat: torch.Tensor, modality: torch.Tensor | None) -> torch.Tensor:
        """按模态应用独立投影；未启用或 modality 缺失时原样返回。"""
        if not self.modality_projection or modality is None:
            return feat
        out = torch.empty_like(feat)
        o_mask = modality == 0
        s_mask = modality == 1
        if o_mask.any():
            out[o_mask] = self.opt_proj(feat[o_mask])
        if s_mask.any():
            out[s_mask] = self.sar_proj(feat[s_mask])
        return out

    def forward(
        self, feat: torch.Tensor, modality: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._project(feat, modality)
        return self.shared_head(feat)

    def set_arcface(self, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        self.shared_head.set_arcface(num_classes, scale=scale, margin=margin)

    def arcface_forward(
        self, feat: torch.Tensor, labels: torch.Tensor, modality: torch.Tensor | None = None
    ) -> torch.Tensor:
        feat = self._project(feat, modality)
        return self.shared_head.arcface_forward(feat, labels)


class DirectionalProjectionHead(nn.Module):
    """方向特异性非对称投影头（Phase 4）。

    在共享 BNNeckHead 输出的检索特征之上，为 O2S 和 S2O 方向各增加一个轻量级
    线性投影，学习方向特定的判别空间：
    - O2S 方向：光学 query 检索 SAR gallery，投影后特征更关注"从光学推断 SAR 散射"的线索
    - S2O 方向：SAR query 检索光学 gallery，投影后特征更关注"从 SAR 散射推断光学外观"的线索

    训练时按 batch 内样本的 modality 路由到对应投影层计算跨模态损失；
    推理时按 query_type 选择对应投影层提取特征。
    """

    def __init__(self, feat_dim: int, proj_dim: int = 0) -> None:
        super().__init__()
        if proj_dim <= 0:
            proj_dim = feat_dim
        # O2S：光学 query → SAR gallery 方向投影
        self.proj_o2s = nn.Linear(feat_dim, proj_dim, bias=False)
        # S2O：SAR query → 光学 gallery 方向投影
        self.proj_s2o = nn.Linear(feat_dim, proj_dim, bias=False)
        self.out_dim = proj_dim

        nn.init.xavier_uniform_(self.proj_o2s.weight)
        nn.init.xavier_uniform_(self.proj_s2o.weight)

    def forward(
        self, feat: torch.Tensor, modality: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (feat_o2s, feat_s2o)：两个方向的投影特征（均已 L2 归一化）。

        训练时调用方按 modality 选择对应投影计算损失；推理时按 query_type 选择。
        """
        feat_o2s = F.normalize(self.proj_o2s(feat), dim=1)
        feat_s2o = F.normalize(self.proj_s2o(feat), dim=1)
        return feat_o2s, feat_s2o

    def extract_directional(
        self, feat: torch.Tensor, direction: str
    ) -> torch.Tensor:
        """按方向提取投影特征。direction: "O2S" 或 "S2O"。"""
        if direction == "O2S":
            return F.normalize(self.proj_o2s(feat), dim=1)
        elif direction == "S2O":
            return F.normalize(self.proj_s2o(feat), dim=1)
        else:
            # O2O 或其他方向：返回原始特征（未投影）
            return F.normalize(feat, dim=1)
