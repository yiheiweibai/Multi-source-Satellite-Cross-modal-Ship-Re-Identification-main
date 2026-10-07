"""全局配置：dataclass 定义 + YAML 加载/保存。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # ---------- 基础 ----------
    seed: int = 42
    device: str = "cuda"            # cuda / cpu，不可用时自动回退
    project_name: str = "ship_reid"

    # ---------- 数据路径 ----------
    # 两种标注来源：
    #   JSON 模式：train_ann / val_ann 指向 {image_path, identity, modality} 标注 json（模拟/自建数据）
    #   CSV 模式（官方 labels.csv）：train_labels_csv / val_labels_csv 指向四列
    #     image_id,ship_id,modality,image_path 的 csv，image_path 相对 csv 所在数据包根目录；
    #     非空时优先于 train_ann / val_ann。
    data_root: str = "data"
    train_ann: str = ""             # JSON 模式标注（默认留空；当前使用 CSV 模式，见 train_labels_csv）
    val_ann: str = ""               # JSON 模式验证标注（默认留空；当前使用 CSV 模式，见 val_labels_csv）
    train_labels_csv: str = ""       # 官方训练 labels.csv（非空则 CSV 模式）
    val_labels_csv: str = ""         # 验证 labels.csv（CSV 模式下为空则跳过验证）
    test_task_json: str = ""         # 初赛/复赛测试包 task.json（inference --task_json 也可指定）
    local_val_task_json: str = ""    # 本地验证任务 json（由 scripts/build_local_val.py 生成）
    local_val_gt_json: str = ""      # 本地验证 GT json（同上）
    max_samples: int = 0             # >0 时每个 CSV 数据集只取前 N 个样本（冒烟/调试用，0=全部）
    auto_num_classes: bool = True    # True 时训练身份数由数据集自动推断，覆盖 num_classes
    dummy_mode: bool = False         # True 时若标注不存在则自动生成模拟数据（默认禁用）

    # ---------- 数据加载 ----------
    image_size: int = 256            # ResNet 骨干输入尺寸；ViT 骨干自动使用 224
    num_workers: int = 4
    sar_speckle_aug: bool = True     # 训练时对 SAR 图模拟斑点噪声增强
    # 模态独立归一化（默认 ImageNet；可由 scripts/compute_stats.py 生成后填入）
    optical_mean: tuple = (0.485, 0.456, 0.406)
    optical_std: tuple = (0.229, 0.224, 0.225)
    sar_mean: tuple = (0.485, 0.456, 0.406)
    sar_std: tuple = (0.229, 0.224, 0.225)
    # SAR 专属增强：禁用 ColorJitter，改用强度抖动 + 弹性形变
    sar_color_jitter: bool = False

    # ---------- PK 采样 ----------
    pk_p: int = 8                    # 每个 batch 的身份数 P
    pk_k: int = 4                    # 每个身份采样 K 张
    pk_prefer_dual_modal: bool = True  # 优先选取同时含光学/SAR 的身份
    pk_sampler: str = "cross_modal"  # 采样器: cross_modal=模态均衡 PK（默认）, pk=普通 PK

    # ---------- 模型 ----------
    backbone: str = "vit_base_patch16_224"   # vit_base_patch16_224（默认）/ resnet50
    pretrained: bool = True
    # 自定义预训练权重路径；由 scripts/download_vit_pretrained.py 生成。
    # 文件存在时离线加载；不存在时回退 timm 在线预训练下载。
    pretrained_path: str = "weights/vit_base_patch16_224.pth"
    share_layer: str = "layer3"      # ResNet 从该层起共享（layer3 / layer4）
    vit_split_layer: int = 6         # ViT 前 N 个 block 为模态特定
    num_classes: int = 40            # 身份分类头输出（auto_num_classes 时由数据集推断）
    embedding_dim: int = 0           # 检索特征维度（0 = 骨干输出维度，不额外投影）
    modality_projection: bool = False  # True 时使用模态感知投影头（O/S 独立投影）

    # ---------- 损失 ----------
    w_supcon: float = 0.5            # 跨模态监督对比损失权重
    w_triplet: float = 0.3           # 难样本三元组损失权重
    w_ce: float = 1.0                # 身份分类损失权重
    w_arcface: float = 0.0           # ArcFace 损失权重（>0 时启用）
    arcface_scale: float = 30.0      # ArcFace scale
    arcface_margin: float = 0.5      # ArcFace margin
    supcon_temperature: float = 0.07
    supcon_base_temperature: float = 0.07
    triplet_margin: float = 0.3
    adaptive_margin: bool = False    # 三元组自适应 margin
    label_smooth: float = 0.1        # 身份分类标签平滑
    # ---------- 跨模态损失（路线 ②：显式跨模态对齐，默认关闭） ----------
    w_cm_infonce: float = 0.0        # 跨模态 InfoNCE 权重（>0 启用）
    w_cm_triplet: float = 0.0        # 跨模态 hard triplet 权重（>0 启用）
    cm_temperature: float = 0.07     # 跨模态 InfoNCE 温度
    cm_margin: float = 0.3           # 跨模态 hard triplet margin

    # ---------- 训练 ----------
    epochs: int = 10
    lr: float = 3e-4                 # 头/分类层学习率
    backbone_lr_scale: float = 0.1   # 骨干（浅层/共享层）相对学习率倍率
    weight_decay: float = 5e-4
    scheduler: str = "cosine"        # cosine / multistep
    total_steps: int = 0             # >0 时使用 OneCycle（按总迭代数）
    warmup_epochs: int = 1
    ema_decay: float = 0.999
    use_ema: bool = True             # 启用 EMA 权重（推理时使用）
    amp: bool = True                 # 启用 AMP 混合精度
    resume: str = ""                 # checkpoint 路径，非空则断点续训

    # ---------- 输出 ----------
    output_dir: str = "outputs"
    log_interval: int = 20
    eval_interval: int = 1
    tb_enabled: bool = True
    save_every: int = 5             # >0 时每 N 个 epoch 另存一份 epoch_XXX.pth；0=关闭

    # ---------- 推理 ----------
    topk: int = 10
    test_batch_size: int = 64        # 推理时特征提取 batch size

    # ---------- 加载/保存 ----------
    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "Config":
        cfg = cls()
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True, sort_keys=False)
