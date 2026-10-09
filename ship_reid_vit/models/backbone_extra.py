"""额外的双分支骨干（ConvNeXt / Swin），与 ``DualBranchResNet`` 同构。

浅层（模态特定）：stem / patch_embed + 前 ``split_idx`` 个 stage（layer），
    光学（optical）与 SAR 各持一份，双分支初始权重一致（均来自预训练）。
深层（模态共享）：剩余 stage（layer）+ 归一化 + 分类头（含全局池化），
    一份权重，强制两模态向共同的结构语义收敛。

契约与 ``DualBranchResNet`` 一致：
    forward(x, modality) -> (B, out_dim)，``modality`` 为 (B,) 张量（0=optical 1=sar）。

实测（timm 1.0.30）：
- ``convnext_tiny``: children = stem / stages(4) / norm_pre(Identity) / head(NormMlpClassifierHead)，
  张量布局全程 NCHW，最终 LayerNorm2d 位于 head 内（池化之后）。
- ``swin_tiny_patch4_window7_224``: children = patch_embed / layers(4) / norm / head(ClassifierHead)，
  张量布局为 NHWC，head 内部按 NHWC 做全局池化；该版本无 ``pos_drop``。
深层共享统一复用 timm 的 ``head``（自带正确布局的全局池化 + 归一化），
经实测与完整模型前向逐位一致。
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .backbone import MODALITY_OPTICAL, MODALITY_SAR, _load_pretrained_weights

logger = logging.getLogger("backbone")


class _DualBranchTimm(nn.Module):
    """ConvNeXt / Swin 双分支共用实现：浅层逐模态、深层共享、按模态分组批处理。

    子类只需实现 ``_split(base, split_idx) -> (shallow_modules, shared_modules, out_dim)``。
    """

    def __init__(
        self,
        backbone_name: str,
        split_idx: int = 2,
        pretrained: bool = True,
        pretrained_path: str = "",
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone_name
        # num_classes=0：分类头退化为 Identity，仅保留全局池化 + 归一化
        base = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        # 本地预训练权重在拆分双分支之前加载到完整 timm 模型，保证键名一一对应
        if pretrained_path and Path(pretrained_path).exists():
            _load_pretrained_weights(base, pretrained_path)
            logger.info(f"{backbone_name} 载入本地预训练权重: {pretrained_path}")

        shallow_modules, shared_modules, out_dim = self._split(base, split_idx)

        self.optical_encoder = nn.Sequential(*shallow_modules)
        # 双分支初始权重一致（均来自预训练），训练中随模态分叉
        self.sar_encoder = copy.deepcopy(self.optical_encoder)
        self.shared = nn.Sequential(*shared_modules)
        self.out_dim = out_dim

    def _split(self, base: nn.Module, split_idx: int):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        # modality: (B,) 0=optical 1=sar；按模态分组批处理，避免逐样本循环
        is_sar = modality == MODALITY_SAR
        opt_idx = (~is_sar).nonzero(as_tuple=True)[0]
        sar_idx = is_sar.nonzero(as_tuple=True)[0]
        out = torch.empty(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        if opt_idx.numel() > 0:
            f = self.optical_encoder(x[opt_idx])
            # AMP 下卷积输出为 fp16，而 out 依输入图像 dtype 为 fp32，需显式对齐
            out[opt_idx] = self.shared(f).flatten(1).to(out.dtype)
        if sar_idx.numel() > 0:
            f = self.sar_encoder(x[sar_idx])
            out[sar_idx] = self.shared(f).flatten(1).to(out.dtype)
        return out


class DualBranchConvNeXt(_DualBranchTimm):
    """ConvNeXt（timm）双分支。

    浅层 = ``stem`` + ``stages[:split_idx]``（每模态一份）；
    深层共享 = ``stages[split_idx:]`` + ``norm_pre`` + ``head``（含全局池化与最终 LayerNorm）。
    """

    def _split(self, base: nn.Module, split_idx: int):
        n_stages = len(base.stages)
        split = max(1, min(int(split_idx), n_stages - 1))
        shallow = [base.stem, *list(base.stages[:split])]
        shared = [*list(base.stages[split:]), base.norm_pre, base.head]
        return shallow, shared, int(base.num_features)


class DualBranchSwin(_DualBranchTimm):
    """Swin（timm）双分支。

    浅层 = ``patch_embed`` (+ ``pos_drop`` 若存在) + ``layers[:split_idx]``（每模态一份）；
    深层共享 = ``layers[split_idx:]`` + ``norm`` + ``head``（含 NHWC 全局池化）。
    """

    def _split(self, base: nn.Module, split_idx: int):
        n_layers = len(base.layers)
        split = max(1, min(int(split_idx), n_layers - 1))
        shallow = [base.patch_embed]
        pos_drop = getattr(base, "pos_drop", None)
        if pos_drop is not None:  # 部分 timm 版本的 Swin 在 patch_embed 后接 pos_drop
            shallow.append(pos_drop)
        shallow += list(base.layers[:split])
        shared = [*list(base.layers[split:]), base.norm, base.head]
        return shallow, shared, int(base.num_features)