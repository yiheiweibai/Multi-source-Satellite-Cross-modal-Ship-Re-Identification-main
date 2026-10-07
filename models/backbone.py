"""双分支骨干网络：浅层模态特定（光学/SAR 各一份）、深层模态共享。

对应技术方案"以船体结构特征为模态不变桥梁"的建模主线：
- 浅层：分别适配光学（纹理/轮廓）与 SAR（散射/边缘）的低层特征
- 深层：共享权重，强制两模态向共同的结构语义收敛
- 支持 ResNet50 与 ViT 两种骨干，均可加载 ImageNet 预训练权重
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

MODALITY_OPTICAL = 0
MODALITY_SAR = 1


class DualBranchResNet(nn.Module):
    """ResNet 双分支。share_layer 之后的层共享（默认 layer3 起共享）。

    结构: [stem+layer1+layer2] 各模态一份（浅层特定）
          [layer3+layer4+avgpool] 一份（深层共享）
    """

    def __init__(self, share_layer: str = "layer3", pretrained: bool = True) -> None:
        super().__init__()
        base = self._build_base(pretrained)

        # 浅层（模态特定）
        shallow = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2
        )
        if share_layer == "layer4":
            shallow = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3,
            )
        self.optical_encoder = shallow
        # 双分支初始权重一致（均来自预训练），训练中随模态分叉
        self.sar_encoder = copy.deepcopy(shallow)

        # 共享深层
        tail = [base.layer3, base.layer4]
        if share_layer == "layer4":
            tail = [base.layer4]
        self.shared = nn.Sequential(*tail, base.avgpool)

        self.out_dim = 2048

    @staticmethod
    def _build_base(pretrained: bool):
        from torchvision import models

        try:  # torchvision >= 0.13
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            return models.resnet50(weights=weights)
        except TypeError:
            return models.resnet50(pretrained=pretrained)

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        # modality: (B,) 0=optical 1=sar；按模态分组批处理，避免逐样本循环
        is_sar = modality == MODALITY_SAR
        opt_idx = (~is_sar).nonzero(as_tuple=True)[0]
        sar_idx = is_sar.nonzero(as_tuple=True)[0]
        out = torch.empty(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        if opt_idx.numel() > 0:
            f = self.optical_encoder(x[opt_idx])
            out[opt_idx] = self.shared(f).flatten(1)
        if sar_idx.numel() > 0:
            f = self.sar_encoder(x[sar_idx])
            out[sar_idx] = self.shared(f).flatten(1)
        return out


class DualBranchViT(nn.Module):
    """ViT 双分支（timm）：patch_embed + 前 split_layer 个 block 为模态特定，
    其余 block + norm 为共享深层。输入尺寸固定 224。
    """

    def __init__(self, split_layer: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        base = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        self.patch_embed = base.patch_embed
        self.pos_drop = base.pos_drop
        self.pos_embed = base.pos_embed

        blocks = list(base.blocks)
        split = max(1, min(split_layer, len(blocks) - 1))
        self.optical_encoder = nn.Sequential(*blocks[:split])
        self.sar_encoder = copy.deepcopy(self.optical_encoder)
        self.shared = nn.Sequential(*blocks[split:], base.norm)

        self.out_dim = base.embed_dim

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        is_sar = modality == MODALITY_SAR
        opt_idx = (~is_sar).nonzero(as_tuple=True)[0]
        sar_idx = is_sar.nonzero(as_tuple=True)[0]
        out = torch.empty(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        if opt_idx.numel() > 0:
            f = self.optical_encoder(x[opt_idx])
            out[opt_idx] = self.shared(f)[:, 0]
        if sar_idx.numel() > 0:
            f = self.sar_encoder(x[sar_idx])
            out[sar_idx] = self.shared(f)[:, 0]
        return out


def _load_pretrained_weights(model: nn.Module, pretrained_path: str) -> None:
    """从自定义路径加载预训练权重（如 SatMAE/RemoteCLIP），自动适配键名。"""
    import logging
    logger = logging.getLogger("backbone")
    state = torch.load(pretrained_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in state.items():
        # 去除常见前缀
        kk = k
        for prefix in ("module.", "backbone.", "encoder."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        if kk in model_state and model_state[kk].shape == v.shape:
            model_state[kk] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    logger.info(f"加载自定义预训练权重: {pretrained_path}，匹配 {loaded}/{len(model_state)} 层，跳过 {len(skipped)} 层")


def build_backbone(cfg) -> nn.Module:
    name = getattr(cfg, "backbone", "resnet50")
    pretrained = getattr(cfg, "pretrained", True)
    pretrained_path = getattr(cfg, "pretrained_path", "")
    if name.startswith("vit"):
        model = DualBranchViT(split_layer=getattr(cfg, "vit_split_layer", 6), pretrained=pretrained)
    else:
        model = DualBranchResNet(
            share_layer=getattr(cfg, "share_layer", "layer3"), pretrained=pretrained
        )
    if pretrained_path:
        _load_pretrained_weights(model, pretrained_path)
    return model
