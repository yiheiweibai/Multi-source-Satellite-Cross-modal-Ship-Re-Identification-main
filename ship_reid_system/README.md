# 赛题06 多源卫星跨模态舰船重识别 · 代码系统框架

面向 **2026 全国大数据与计算智能挑战赛 赛题06「多源卫星跨模态舰船重识别」** 的可运行工程化代码框架。

> **双路线并行**：本仓库自研框架 + TransOSS（ICCV 2025 SOTA）微调。两路推理均叠加 TTA/K-reciprocal/Gallery 聚类后处理，用本地验证协议对比增益。

## 1. 项目结构

```
ship_reid_system/
├── config/
│   ├── config.py            # Config dataclass（全量参数 + yaml 加载/保存）
│   ├── default.yaml         # 默认配置
│   └── train_gpu.yaml       # GPU 正式训练配置（RTX 4070 Ti SUPER）
├── data/
│   ├── dataset.py           # 光学/SAR 数据集（JSON + 官方 CSV 两种模式）
│   ├── sampler.py           # PK 采样器（P 身份 × K 图，优先跨模态身份）
│   ├── transforms.py        # 模态独立归一化 + SAR 专属增强 + 斑点噪声模拟
│   ├── test_dataset.py      # 测试集 Query/Gallery 数据集 + task.json 解析
│   └── dummy.py             # 模拟数据生成器（数据开放前跑通全流程）
├── models/
│   ├── backbone.py          # 双分支骨干：ResNet50 / ViT（浅层特定 + 深层共享）
│   ├── heads.py             # BNNeck 头 + 可选投影 + ArcFace 分类头
│   └── model.py             # ShipReIDModel 组装
├── losses/
│   ├── supcon_loss.py       # 跨模态监督对比损失（SupCon）
│   ├── triplet_loss.py      # 三元组损失（难样本挖掘/自适应 margin）
│   └── composed_loss.py     # SupCon + Triplet + CE + ArcFace 加权组合
├── trainers/
│   ├── trainer.py           # 训练循环 + EMA + AMP + checkpoint 保存/恢复
│   └── lr_scheduler.py      # 分层学习率 + Warmup + Cosine 调度
├── inference/
│   ├── retrieval.py         # 特征提取（支持 TTA）+ EMA 权重加载
│   ├── submission.py        # 提交格式生成 + 模态过滤 + 格式自检
│   └── postprocess.py       # k-reciprocal 重排序 + QE + Gallery 聚类
├── utils/
│   ├── metrics.py           # mAP / R@K（整体 + O2S/S2O/O2O 分方向）
│   └── logger.py            # 日志封装
├── scripts/
│   ├── build_local_val.py   # 生成本地验证集（task.json + gt.json）
│   ├── multi_split_val.py   # 多次随机划分验证，取均值±标准差
│   ├── compute_stats.py     # 计算训练集各模态 mean/std
│   ├── pseudo_label.py      # 伪标签自训练
│   ├── transoss_prepare_data.py  # 训练数据→TransOSS bounding_box_train 格式
│   └── transoss_inference.py     # TransOSS 推理 + 后处理 + 提交（在 Hoss-ReID 内运行）
├── transoss_config/
│   └── hoss_transoss_competition.yml  # TransOSS 微调配置（指向赛题数据）
├── outputs/                 # 训练产物（自动创建）
│   ├── logs/                # tensorboard
│   └── checkpoints/         # last.pth / best.pth
├── setup.bat / setup_env.ps1  # 一键环境安装（CUDA 12.4）
├── train.bat                # GPU 训练
├── inference.bat            # 生成提交 prediction.json
├── local_val.bat            # 本地验证集预测 + 评分
├── pseudo_label.bat         # 伪标签自训练
├── run_all.bat              # 一键全流程
├── requirements.txt
├── train.py / inference.py / evaluate.py
└── README.md
```

## 2. 环境安装

**推荐（Windows GPU 一键安装）**：双击 `setup.bat`，或执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
```

脚本自动创建 `.venv` → 安装 CUDA 12.4 版 PyTorch → 安装依赖 → 验证 GPU。

**手动安装**：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

- 驱动建议 ≥ 551.x（CUDA 12.4）
- CPU 兜底：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## 3. 数据放置约定

目录结构（`赛题6-初赛/` 与 `ship_reid_system/` 同级）：

```
项目根目录/
├── ship_reid_system/
└── 赛题6-初赛/
    ├── 训练数据/
    │   ├── optical/         # 光学训练图像
    │   ├── sar/             # SAR 训练图像
    │   └── labels.csv       # image_id,ship_id,modality,image_path
    └── 初赛测试数据/
        ├── query/
        ├── gallery/
        └── task.json
```

## 4. 快速开始

### 4.1 一键全流程

双击 `run_all.bat`：生成本地验证集 → GPU 训练 → 本地验证评测 → 生成提交文件。

### 4.2 分步执行

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 环境安装 | `setup.bat` | 安装 CUDA 12.4 环境 |
| 训练 | `train.bat` | GPU 训练（EMA + AMP + Warmup） |
| 本地验证 | `local_val.bat` | 本地验证集预测 + 评分 |
| 生成提交 | `inference.bat` | 生成 `prediction.json`（TTA+重排序+QE+聚类） |
| 伪标签自训练 | `pseudo_label.bat` | gallery 伪标签 → 扩展训练 → 重训 |

## 5. 核心功能

### 5.1 模型

| 特性 | 说明 | 配置 |
|---|---|---|
| 双分支骨干 | 浅层模态特定 + 深层共享 | `backbone`, `share_layer` |
| BNNeck 头 | 检索/分类解耦 | — |
| 可配置嵌入维度 | 可选投影到指定维度 | `embedding_dim`（0=不投影） |
| 遥感预训练 | 支持 SatMAE/RemoteCLIP 权重 | `pretrained_path` |
| ArcFace 损失 | 角度 margin 分类 | `w_arcface`（>0 启用） |

### 5.2 训练

| 特性 | 说明 | 配置 |
|---|---|---|
| 组合损失 | SupCon + Triplet + CE + ArcFace | `w_*` |
| PK 采样 | 优先跨模态身份，不足 K 时重复+增强 | `pk_p`, `pk_k` |
| EMA 权重 | 推理时优先用 EMA 权重 | `use_ema`, `ema_decay` |
| AMP 混合精度 | autocast + GradScaler | `amp` |
| Warmup + Cosine | 前 N epoch 线性升温 | `warmup_epochs` |
| 标签平滑 | CE 损失标签平滑 | `label_smooth` |

### 5.3 数据增强

| 特性 | 说明 | 配置 |
|---|---|---|
| 模态独立归一化 | optical/SAR 各自 mean/std | `optical_mean/std`, `sar_mean/std` |
| SAR 专属增强 | 禁用 ColorJitter，改用强度抖动 | `sar_color_jitter: false` |
| 光学增强 | ColorJitter + GaussianBlur + RandomErasing | — |
| SAR 斑点噪声 | Gamma 乘性噪声模拟 | `sar_speckle_aug` |

### 5.4 推理后处理

| 特性 | CLI | 说明 |
|---|---|---|
| TTA | `--tta` | 水平翻转特征平均 |
| K-reciprocal 重排序 | `--rerank` | Jaccard 相似度重排 |
| Query Expansion | `--qe` | Top-1 gallery 特征扩展 query |
| Gallery 聚类 | `--cluster` | DBSCAN 聚类 + 类别中心检索 |

可组合使用：`--tta --rerank --qe --cluster`

### 5.5 验证

| 脚本 | 说明 |
|---|---|
| `scripts/build_local_val.py` | 划分训练/验证集，生成 task.json + gt.json |
| `scripts/multi_split_val.py` | 多次随机划分，取均值±标准差 |
| `evaluate.py --submission` | 按 O2S/S2O/O2O 计算 R@1 + mAP@10 |

## 6. 命令行用法

### 6.1 训练

```bash
# GPU 正式训练
python train.py --config config/train_gpu.yaml

# 覆盖参数
python train.py --config config/train_gpu.yaml --epochs 120

# 断点续训
python train.py --resume outputs/checkpoints/last.pth
```

### 6.2 生成提交

```bash
# 全后处理（推荐）
python inference.py --config config/train_gpu.yaml \
    --ckpt outputs/checkpoints/best.pth \
    --task_json "../赛题6-初赛/初赛测试数据/task.json" \
    --out_prediction outputs/prediction.json \
    --tta --rerank --qe --cluster

# 仅基线（无后处理）
python inference.py --ckpt outputs/checkpoints/best.pth \
    --task_json "../赛题6-初赛/初赛测试数据/task.json"
```

### 6.3 本地验证

```bash
# 1. 生成本地验证集
python scripts/build_local_val.py \
    --csv "../赛题6-初赛/训练数据/labels.csv" \
    --out_dir "../赛题6-初赛/训练数据"

# 2. 生成验证集预测
python inference.py --ckpt outputs/checkpoints/best.pth \
    --task_json "../赛题6-初赛/训练数据/local_val_task.json" \
    --out_prediction outputs/local_val_pred.json \
    --tta --rerank --qe --cluster

# 3. 评分
python evaluate.py --submission \
    --prediction outputs/local_val_pred.json \
    --task "../赛题6-初赛/训练数据/local_val_task.json" \
    --gt "../赛题6-初赛/训练数据/local_val_gt.json"
```

### 6.4 多次划分验证

```bash
python scripts/multi_split_val.py \
    --ckpt outputs/checkpoints/best.pth \
    --csv "../赛题6-初赛/训练数据/labels.csv" \
    --num_splits 5 --tta --rerank --qe --cluster
```

### 6.5 计算模态统计量

```bash
python scripts/compute_stats.py --csv "../赛题6-初赛/训练数据/labels_train.csv"
```
将输出填入 `config/default.yaml` 的 `optical_mean/std`、`sar_mean/std`。

### 6.6 伪标签自训练

```bash
python scripts/pseudo_label.py \
    --ckpt outputs/checkpoints/best.pth \
    --train_csv "../赛题6-初赛/训练数据/labels_train.csv" \
    --task_json "../赛题6-初赛/初赛测试数据/task.json" \
    --out_csv "../赛题6-初赛/训练数据/labels_pseudo.csv" \
    --threshold 0.85

# 用伪标签扩展数据重训
python train.py --config config/train_gpu.yaml \
    train_labels_csv="../赛题6-初赛/训练数据/labels_pseudo.csv"
```

## 7. 评测指标

赛题评分：R@1 + mAP@10 取平均，综合得分 = 0.45×O2S + 0.45×S2O + 0.10×O2O。

`evaluate.py --submission` 自动计算三方向指标并输出综合得分。

## 8. TransOSS 路线（ICCV 2025 SOTA）

TransOSS 是该赛题同源数据集 HOSS ReID 的官方基线，已在大规模光学-SAR 图像对上做过对比预训练，跨模态对齐能力远强于通用预训练。

### 8.1 克隆仓库

```bash
# 在本项目根目录同级克隆
git clone https://github.com/Alioth2000/Hoss-ReID.git
```

### 8.2 下载预训练权重

从 [HuggingFace](https://huggingface.co/Alioth2000/TransOSS/tree/main) 下载 `vit_b512_pre.pth`，放到 `Hoss-ReID/weights/`。

### 8.3 数据转换

将赛题训练数据转为 TransOSS 的 `bounding_box_train` 格式：

```bash
# 在 ship_reid_system/ 目录执行
python scripts/transoss_prepare_data.py \
    --csv "../赛题6-初赛/训练数据/labels_train.csv" \
    --out_dir "../transoss_data/HOSS/bounding_box_train"
```

### 8.4 配置与微调

将 `transoss_config/hoss_transoss_competition.yml` 复制到 `Hoss-ReID/configs/`，然后：

```bash
cd Hoss-ReID
python train.py --config_file configs/hoss_transoss_competition.yml
```

### 8.5 推理 + 后处理 + 提交

将 `scripts/transoss_inference.py` 复制到 `Hoss-ReID/` 根目录，执行：

```bash
python transoss_inference.py \
    --config_file configs/hoss_transoss_competition.yml \
    --weight logs/competition_transoss/transformer_200.pth \
    --task_json "../赛题6-初赛/初赛测试数据/task.json" \
    --out_prediction ../prediction_transoss.json \
    --tta --rerank --qe --cluster
```

后处理（TTA/k-reciprocal/QE/Gallery聚类）与自研框架完全一致，可在同一份本地验证集上公平对比两路增益。

### 8.6 一键脚本

双击 `transoss_pipeline.bat` 按提示完成数据转换 → 微调 → 推理全流程。

## 9. 常见问题

| 现象 | 排查 |
|---|---|
| `cuda.is_available()` 为 False | 驱动 ≥ 551.x；确认 torch 为 cu124 版；激活 `.venv` |
| 显存 OOM | 降低 `pk_p`/`pk_k`（如 8×4→6×4） |
| 训练慢 | `num_workers` 降为 4；检查 GPU 占用 |
| AMP 报错 | 设 `amp: false` 关闭混合精度 |
| EMA 权重缺失 | 旧 checkpoint 无 `ema_state`，自动回退普通权重 |
| 聚类无效果 | 调整 `eps`（postprocess.py 中，默认 0.5） |
| TransOSS 权重加载失败 | 确认 `PRETRAIN_PATH` 指向 `vit_b512_pre.pth`，且 `TRANSFORMER_TYPE: 'vit_base_patch16_224_TransOSS'` |
