# Phase 1 冒烟测试：验证多尺度特征提取代码路径
# 不加载真实 checkpoint，只用随机初始化的模型验证前向传播

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ship_reid_vit"))

import torch
from config import Config
from models import build_model


def test_multiscale():
    """用随机初始化模型测试多尺度特征提取。"""
    # 创建最小配置
    class Cfg:
        backbone = "vit_base_patch16_224"
        pretrained = False
        pretrained_path = ""
        vit_split_layer = 6
        image_size = 224
        embedding_dim = 0
        num_classes = 100
        modality_projection = False
        w_arcface = 0.0
        arcface_scale = 30.0
        arcface_margin = 0.5

    cfg = Cfg()
    model = build_model(cfg)
    model.eval()

    # 模拟输入 (B=2, C=3, H=224, W=224)
    x = torch.randn(2, 3, 224, 224)
    modality = torch.tensor([0, 1])  # 一个光学，一个SAR

    # 测试全局特征
    with torch.no_grad():
        f_global = model.extract_feature(x, modality)
    print(f"全局特征: {f_global.shape}, 范数: {f_global.norm(dim=1)}")

    # 测试多尺度特征
    with torch.no_grad():
        f_ms = model.extract_multiscale_feature(
            x, modality,
            global_weight=0.5, center_weight=0.3, stripe_weight=0.2,
        )
    print(f"多尺度特征: {f_ms.shape}, 范数: {f_ms.norm(dim=1)}")

    # 验证输出是 L2 归一化的
    assert torch.allclose(f_ms.norm(dim=1), torch.ones(2), atol=1e-5), "多尺度特征未归一化"
    assert f_ms.shape == f_global.shape, "多尺度特征维度与全局特征不一致"

    print("✓ 多尺度特征提取冒烟测试通过")


def test_sar_scatter_aug():
    """测试 SAR 散射点感知增强。"""
    from data.transforms import SARScatterAug

    aug = SARScatterAug(contrast_range=0.4, noise_level_range=(0.05, 0.20), deform_alpha=8.0, p=1.0)
    img = torch.rand(3, 64, 64)
    out = aug(img)
    print(f"SAR 增强: 输入 {img.shape} -> 输出 {out.shape}, 范围 [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == img.shape, "增强后形状改变"
    assert out.min() >= 0 and out.max() <= 1, "像素值超出 [0,1] 范围"
    print("✓ SAR 散射点感知增强冒烟测试通过")


if __name__ == "__main__":
    test_multiscale()
    test_sar_scatter_aug()
    print("\n所有冒烟测试通过！")
