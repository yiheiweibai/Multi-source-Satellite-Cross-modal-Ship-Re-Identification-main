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
│   ├── transoss_inference.py     # TransOSS 推理 + 后处理 + 提交（在 Hoss-ReID 内运行）
│   ├── transoss_checkpoint_fusion.py  # 多 checkpoint 批量验证 + RRF 排名融合
│   ├── transoss_ablation.py      # 一键消融验证（base/preprocess/colormap/tta/rerankqe/mixed 六组合对比，含 ΔBaseline）
│   └── transoss_setup.py         # TransOSS 一键准备（克隆仓库/依赖/数据/占位目录/配置/权重检查）
│   ├── sdfnet_download_weights.py    # SDF-Net 官方权重下载（HuggingFace Chenfree233/SDF-Net）
│   ├── sdfnet_prepare_data.py        # 赛题数据 → HOSS 结构（bounding_box_train/test/query，复用 local_val_task.json）
│   ├── sdfnet_inference.py           # SDF-Net 推理 + 按 query_type 分组后处理 + 提交（在 SDF-Net 内运行）
│   └── fuse_sdfnet_transoss.py       # SDF-Net × TransOSS 跨模型分数融合（weighted / RRF）
├── transoss_config/
│   └── hoss_transoss_competition.yml  # TransOSS 微调配置（指向赛题数据）
├── outputs/                 # 训练产物（自动创建）
│   ├── logs/                # tensorboard
│   └── checkpoints/         # last.pth / best.pth
├── 01_setup.bat / setup_env.ps1  # 一键环境安装（CUDA 12.4）
├── 02_train.bat                # GPU 训练
├── 03_local_val.bat            # 本地验证集预测 + 评分
├── 04_inference.bat            # 生成提交 prediction.json
├── 05_pseudo_label.bat         # 伪标签自训练
├── 06_run_all.bat              # 一键全流程
├── 07_transoss_pipeline.bat    # TransOSS 微调+推理全流程（自动）
├── 08_transoss_setup.bat       # TransOSS 环境一键准备（克隆/依赖/数据/占位目录/配置/权重检查）
├── 09_local_val_transoss.bat   # TransOSS 本地验证一键（备份/划分20%验证集/推理/评测）
├── 10_transoss_fusion.bat      # 多 checkpoint 选优 + RRF 融合一键
├── 11_transoss_ab.bat          # 一键消融验证（对齐路线图 4.1 七组合矩阵，base 只跑一次）
├── 12_sdfnet_setup.bat         # SDF-Net 环境一键准备（venv-sdfnet/依赖/克隆/权重/数据转 HOSS）
├── 12_sdfnet_zero_shot.bat     # SDF-Net 零训练初判（官方 HOSS 权重直接本地验证 base/rerankqe）
├── 12_sdfnet_finetune.bat      # SDF-Net 短 epoch 微调（默认 60 可配）后本地验证
├── 12_sdfnet_fusion.bat        # SDF-Net × TransOSS 跨模型融合一键（weighted / RRF）
├── requirements.txt
├── train.py / inference.py / evaluate.py
└── README.md

experiments/                  # 实验记录体系（对齐路线图 2.2，统一 CSV 登记 + ΔBaseline）
├── README.md                 # 记录规范：必记 O2S/S2O/O2O/Overall 与 ΔBaseline、单变量原则、判定标准
├── exp_001_transoss_base.csv # TransOSS baseline（两次划分 + Public，注明划分批次差异）
├── exp_002_ckpt_fusion.csv   # 10 checkpoint 评测表 + RRF(top3) 结论（无实质增益）
└── exp_003_sar_preprocess_ab.csv  # 11 脚本六组合消融（base/preprocess/colormap/tta/rerankqe/mixed，待 11 跑完填写）
```

## 2. 环境安装

**推荐（Windows GPU 一键安装）**：双击 `01_setup.bat`，或执行：

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
- `requirements.txt` 已含 TransOSS 依赖 `einops`、`yacs`；`08_transoss_setup.bat` 也会幂等补装
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

> 当前 `config/default.yaml` 与 `config/train_gpu.yaml` 均已指向真实数据路径（`../赛题6-初赛/训练数据`、`../赛题6-初赛/初赛测试数据`），CSV 模式优先；`train_ann` / `val_ann` 留空，仅在切换 JSON 标注模式时填写。

## 4. 快速开始

### 4.1 一键全流程

双击 `06_run_all.bat`：生成本地验证集 → GPU 训练 → 本地验证评测 → 生成提交文件。

### 4.2 分步执行

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 环境安装 | `01_setup.bat` | 安装 CUDA 12.4 环境 |
| 训练 | `02_train.bat` | GPU 训练（EMA + AMP + Warmup） |
| 本地验证 | `03_local_val.bat` | 本地验证集预测 + 评分 |
| 生成提交 | `04_inference.bat` | 生成 `prediction.json`（TTA+重排序+QE+聚类） |
| 伪标签自训练 | `05_pseudo_label.bat` | gallery 伪标签 → 扩展训练 → 重训 |
| 一键全流程 | `06_run_all.bat` | 构建验证集 → 训练 → 评测 → 提交 |
| TransOSS 环境准备 | `08_transoss_setup.bat` | 克隆 Hoss-ReID、装依赖、数据转换、建占位目录、生成配置、检查权重 |
| TransOSS 全流程 | `07_transoss_pipeline.bat` | 数据转换 → 微调（自动）→ 推理 → 提交 |
| TransOSS 本地验证 | `09_local_val_transoss.bat` | 备份/划分 20% 验证集/推理/评测 |
| TransOSS 多checkpoint融合 | `10_transoss_fusion.bat` | 逐个权重批量验证 + 选优 RRF 融合 |
| TransOSS 一键消融验证 | `11_transoss_ab.bat` | 同一验证集跑 base/preprocess/rerank+QE/mixed 四组合并评测对比 |
| SDF-Net 环境准备 | `12_sdfnet_setup.bat` | 创建 `.venv-sdfnet`、装 PyTorch 2.2.2+cu118 依赖、克隆 SDF-Net、下载官方权重、数据转 HOSS |
| SDF-Net 零训练初判 | `12_sdfnet_zero_shot.bat` | 官方 HOSS 预训练权重直接本地验证（base / rerank+QE），零训练成本 |
| SDF-Net 短 epoch 微调 | `12_sdfnet_finetune.bat` | 默认 60 epoch（`SDF_EPOCHS` 可配）从官方权重续微调，控制训练时间 |
| SDF-Net × TransOSS 融合 | `12_sdfnet_fusion.bat` | 加权平均 / RRF 跨模型分数融合 + 本地评测对比 |

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

### 8.1 一键准备（推荐）

双击 `08_transoss_setup.bat`（或 `python scripts/transoss_setup.py`），自动完成：
- 克隆 `Hoss-ReID` 到 `ship_reid_system/Hoss-ReID`（已存在则跳过）
- 幂等安装 TransOSS 额外依赖 `einops`、`yacs`
- 赛题训练数据 → `bounding_box_train` 格式转换
- 创建评估占位目录 `query/`、`bounding_box_test/`，避免训练 `EVAL_PERIOD` 阶段崩溃
- 生成 `configs/hoss_transoss_competition.yml`（`ROOT_DIR` 自动替换为机器绝对路径）
- 复制推理脚本并检查预训练权重是否就位

### 8.2 预训练权重

从 [HuggingFace](https://huggingface.co/Alioth2000/TransOSS/tree/main) 下载 `vit_b512_pre.pth`，放到 `Hoss-ReID/weights/`。`08` 脚本会在缺失时提示。

### 8.3 一键微调 + 推理（推荐）

双击 `07_transoss_pipeline.bat`，自动执行：数据转换（幂等）→ 微调 200 epochs → 推理生成 `prediction_transoss.json`。

### 8.4 手动执行（可选）

```bash
# 数据转换
python scripts/transoss_prepare_data.py \
    --csv "../赛题6-初赛/训练数据/labels_train.csv" \
    --out_dir "../transoss_data/HOSS/bounding_box_train"

# 微调
cd Hoss-ReID
python train.py --config_file configs/hoss_transoss_competition.yml

# 推理
python transoss_inference.py \
    --config_file configs/hoss_transoss_competition.yml \
    --weight logs/competition_transoss/transformer_200.pth \
    --task_json "../赛题6-初赛/初赛测试数据/task.json" \
    --out_prediction ../prediction_transoss.json
```

### 8.5 后处理与评估说明

- 推理脚本**默认关闭后处理**（直接余弦相似度检索）。如需叠加，加 `--tta --rerank --qe --cluster`；后处理参数建议先在自研路线的本地验证集上调优，再平移到 TransOSS。
- 后处理按候选模态分组执行，并排除 query 自身（修复 O2O 方向 R@1 偏低与跨模态污染问题）。
- **训练内置评估不可信**：评估集为占位数据、与训练集身份重叠，mAP/R@1 虚高。以测试集推理 + 本地验证（`evaluate.py --submission`）为准。

### 8.6 本地验证（一键）

双击 `09_local_val_transoss.bat`，一键完成 TransOSS 本地验证：

1. 备份 `labels_train.csv` → `labels_train.backup.csv`（首次运行自动执行，已有备份则跳过）
2. `build_local_val.py` 从全量标签划分 20% 身份作为验证集，生成 `local_val_task.json` / `local_val_gt.json`，并重写 `labels_train.csv` 为 80% 身份子集
3. `transoss_inference.py` 在 Hoss-ReID 仓库内对验证集推理（默认无后处理）
4. `evaluate.py --submission` 输出 O2S / S2O / O2O 与综合得分，可对照自研基线（无后处理综合 0.1162）

**运行前提**：已跑通 `08_transoss_setup.bat`（环境准备）与 `07_transoss_pipeline.bat`（训练权重 `logs/competition_transoss/transformer_200.pth`），且赛题数据 `labels.csv` 已同步到 `../赛题6-初赛/训练数据/`。

**使用方式**：在目标机（16GB 显存、已跑通 07 的机器）双击运行，全程自动，结果自动与自研基线对比。

**注意事项**：
- 脚本会重新划分验证集并重写 `labels_train.csv`，首次运行前自动备份至 `labels_train.backup.csv`，请勿手动删除备份
- 默认**无后处理**（直接余弦相似度检索）；如需重测后处理增益，在脚本第 3 步推理命令追加 `--tta --rerank --qe --cluster` 后重跑

### 8.7 多 checkpoint 选优与 RRF 融合

**用途**：`07_transoss_pipeline.bat` 微调结束后，`Hoss-ReID/logs/competition_transoss/` 下会保留多个 `transformer_*.pth`（如 160/180/200）。逐个在本地验证集上推理评测后，选取综合得分最高的 top_k 个 checkpoint，用 RRF（Reciprocal Rank Fusion）融合它们的排名，通常比单一末轮权重更稳。

**用法**：双击 `10_transoss_fusion.bat`（或直接执行）

```bash
.venv\Scripts\python.exe scripts\transoss_checkpoint_fusion.py ^
    --task_json "..\赛题6-初赛\训练数据\local_val_task.json"
```

可选参数：`--top_k 3`（融合 checkpoint 数，默认 3）、`--preprocess`（融合时同步启用 SAR 预处理）、`--weight_dir` / `--out_dir` 自定义权重与输出目录。

**流程**：glob 排序 `transformer_*.pth` → 逐个加载权重提取 query/gallery 特征（复用 `transoss_inference.extract_features`，按 query_type 过滤候选模态并排除 query 自身）→ 写 `pred_ckpt_<epoch>.json` → 子进程 `evaluate.py --submission` 评测（O2S/S2O/O2O 与综合）→ 按综合得分取 top_k → RRF 融合（`score = Σ 1/(60+rank)`，未进该 query 前 10 则计 0）→ 每个 query 在候选集按融合分降序取前 10，写 `pred_fusion.json` 并评测，最后打印对比汇总表。

**预期**：若各 epoch 权重综合得分接近（通常 0.10~0.13），融合结果一般不低于最优单权重；本地验证用融合分数指导最终提交权重的选择。

### 8.8 SAR 预处理增强（--preprocess）

**用途**：SAR 图像天然带相干斑噪声、对比度低。`transoss_inference.py` 与 `transoss_checkpoint_fusion.py` 均支持 `--preprocess`，在 transform 前对 modality=1（SAR）图像做增强，可提升 SAR 分支特征质量。

**流程**：转灰度 → `cv2.fastNlMeansDenoising(h=10)` 去斑 → `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` 对比度增强 → 默认复制为 3 通道 RGB；加 `--sar_colormap` 则 `cv2.applyColorMap(JET)` 伪彩转 RGB。`cv2` 缺失时打印警告并跳过增强，不影响原逻辑（默认关闭）。

**用法**（本地验证/提交时追加）：

```bash
python transoss_inference.py --config_file configs/hoss_transoss_competition.yml ^
    --weight logs/competition_transoss/transformer_200.pth ^
    --task ../赛题6-初赛/训练数据/local_val_task.json ^
    --out_prediction ../prediction_transoss_preprocess.json --preprocess
```

**预期**：SAR 去斑 + CLAHE 可缓解相干斑噪声对检索的干扰，O2S/S2O 方向可能有小幅增益；`--sar_colormap` 伪彩是否增益需在本地验证集上对比确认（建议先 `--preprocess` 测基线，再叠加 `--sar_colormap` 对比）。

### 8.9 一键消融验证（11_transoss_ab.bat）

**用途**：对齐路线图 4.1/7.1/7.2 的实验矩阵，一次性验证 SAR 预处理、排序后处理与两者混合是否带来正增益。`11_transoss_ab.bat` 基于**同一份** `local_val_task.json` 依次跑 6 个组合（base 只跑一次）并评测，保证横向可比：1) `base`（无增强，baseline 参照）；2) `preprocess`（`--preprocess`，SAR 去斑 + CLAHE）；3) `preprocess_colormap`（`--preprocess --sar_colormap`，SAR 预处理 + JET 伪彩）；4) `preprocess_tta`（`--preprocess --tta`，SAR 预处理 + 水平翻转 TTA）；5) `rerankqe`（`--rerank --qe`，k-reciprocal rerank + QE，k1=20 k2=6 lambda=0.3 固定）；6) `mixed`（`--preprocess --rerank --qe`，预处理 + 排序后处理混合）。

**用法**：双击 `11_transoss_ab.bat`（或直接执行）

```bash
.venv\Scripts\python.exe scripts\transoss_ablation.py ^
    --task_json "../赛题6-初赛/训练数据/local_val_task.json"
```

**验证集策略**：脚本先检查 `local_val_task.json` 是否已存在——存在则直接复用（不重新划分，避免覆盖 09 生成的验证集）；不存在才调用 `build_local_val.py` 生成一次（此时会重写 `labels_train.csv` 为 80% 身份子集，首次运行建议先备份为 `labels_train.backup.csv`）。

**流程**：6 个组合按序调用 `transoss_inference.py`（同一权重 `transformer_200.pth`，各自输出独立预测 `pred_ab_base.json` / `pred_ab_preprocess.json` / `pred_ab_preprocess_colormap.json` / `pred_ab_preprocess_tta.json` / `pred_ab_rerankqe.json` / `pred_ab_mixed.json`，互不覆盖）→ 每个组合子进程 `evaluate.py --submission` 评测（O2S/S2O/O2O 方向得分与综合得分）→ 末尾打印横向对比汇总表（含 **ΔBaseline** 列，相对 base 综合得分）。

**实验登记**：11 脚本跑完后，将六组合的 O2S/S2O/O2O/Overall/ΔBaseline 回填 `experiments\exp_003_sar_preprocess_ab.csv`（模板已就位）；所有已跑实验统一登记在 `experiments\` 目录（对齐路线图 2.2，见文件树与各 CSV 备注）。

**建议**：只保留汇总表中相对 `base` 为**正增益**的组合项，将正增益项叠加进最终推理配置；负增益项直接弃用。

### 8.10 一键脚本

| 脚本 | 作用 |
|---|---|
| `08_transoss_setup.bat` | 环境一键准备（克隆/依赖/数据/占位目录/配置/权重检查） |
| `07_transoss_pipeline.bat` | 微调 + 推理全流程（自动） |
| `09_local_val_transoss.bat` | 本地验证（备份/划分 20% 验证集/推理/评测） |
| `10_transoss_fusion.bat` | 多 checkpoint 选优 + RRF 融合 |
| `11_transoss_ab.bat` | 一键消融验证（对齐路线图 4.1 七组合：base / preprocess / colormap / tta / rerankqe / mixed，含 ΔBaseline 对比） |
| `12_sdfnet_setup.bat` | SDF-Net 环境一键准备（venv-sdfnet / 依赖 / 克隆 / 权重下载 / 数据转 HOSS） |
| `12_sdfnet_zero_shot.bat` | SDF-Net 零训练初判（官方 HOSS 权重 base / rerank+QE 本地验证） |
| `12_sdfnet_finetune.bat` | SDF-Net 短 epoch 微调（默认 60 可配，从官方权重续训）+ 本地验证 |
| `12_sdfnet_fusion.bat` | SDF-Net × TransOSS 跨模型融合（weighted / RRF，可配权重） |

### 8.11 优化节奏建议（零训练成本优先）

优化遵循"先吃零成本推理端增益、不达标再重训"的节奏，避免盲目投入高成本训练：

1. **零训练成本阶段（约半小时，预计 +4~8 分）**：先运行 `ship_reid_system\10_transoss_fusion.bat` 看多 checkpoint 选优与 RRF 融合的对比汇总表，再依次尝试 `--preprocess`（SAR 预处理）、TTA、rerank、QE 等推理端增强；**每步只保留带来正增益的项**，逐项叠加吃干净零训练成本的红利。
2. **达标即提交**：本地验证分数达到目标后，直接对测试集执行推理并提交，不再追加改动。
3. **不达标再重训（高成本兜底）**：仍不足时，才考虑需要重训的高成本项，如伪标签自训练（1 轮约 3.9 小时）、分辨率微调等；重训**可基于已有 `transformer_*.pth` 续训**，避免从头开始的时间开销。

### 8.12 当前进度（实验闭环阶段）

**阶段定位**：当前处于路线图**阶段 1~3 实验闭环**——以零训练成本增益优先（推理端 SAR 预处理 / TTA / rerank / QE / checkpoint 选优 / RRF），**尚未进入重训阶段**；自研 CrossModal 模型（`w_cm_infonce` / `w_cm_triplet`）仍在关闭状态，后续按路线图第 15 节 ⑧⑨ 再启用。

**已完成结果摘要**（对照 `experiments\` 台账，记录规范见 `experiments\README.md`）：

| 实验 | 结果 | 结论 |
|---|---|---|
| TransOSS 全流程 + 测试集提交 | `prediction_transoss.json` 已提交，官方 **Public = 0.5055** | 全流程跑通 |
| 本地验证 base | 划分1：**0.4881**（O2S 0.4612 / S2O 0.4126 / O2O 0.9490）；划分2：**0.4831**（O2S 0.4637 / S2O 0.3990 / O2O 0.9492） | 两次划分存在批次差异，Δ 不可直接对比（详见 `exp_001`） |
| 10 checkpoint 选优 | **200 = 0.4831 最优**；RRF(top3) = 0.4835 | RRF 相对最优单点 **Δ≈0.0004，在噪声内、无实质增益**，暂不采用（详见 `exp_002`） |
| 自研基线对比 | 自研 0.1162 → TransOSS 0.4831 | TransOSS 相对自研提升约 **4.2 倍** |

**待执行**：`11_transoss_ab.bat` 六组合消融（base / preprocess / preprocess+colormap / preprocess+tta / rerankqe / mixed，同一 `local_val_task.json` 横向对比，含 ΔBaseline 汇总表）；跑完将 O2S/S2O/O2O/Overall/ΔBaseline 回填 `experiments\exp_003_sar_preprocess_ab.csv`（模板已就位），据此决定哪些推理端增强纳入最终提交。

**决策原则**：只保留相对 base 综合得分 **增益 > 0.005** 的项；本地验证集是实验裁判、Public 是最终裁判，两者分布不同禁止直接互比 Δ。

### 8.13 SDF-Net 接入与使用说明

**背景**：SDF-Net（Structure-aware Disentangled Feature Learning）是面向光学-SAR 舰船 ReID 的 SOTA 方法（基于 ViT + 结构一致性 + 解耦特征学习），在 HOSS-ReID 上全面优于 TransOSS（All mAP 60.9 vs 57.4，R1 69.9 vs 65.9）。官方实现与预训练权重公开，按赛题规则属允许的外部资源（已在 8.17 完整披露）。

**来源与布局**：
- 代码：`https://github.com/cfrfree/SDF-Net`（官方克隆至 `ship_reid_system/SDF-Net`，commit `15cbb1a88742256c9a6754d4f29f956991366f93`）
- 权重：`https://huggingface.co/Chenfree233/SDF-Net` → `SDF-Net.pth`（约 333MB，基于 HOSS-ReID 预训练）
- 第三方源码不并入主仓库（`ship_reid_system/SDF-Net/` 已在 `.gitignore`），复现时按上述地址克隆即可

**数据（HOSS 结构）**：`12_sdfnet_setup.bat` 通过 `scripts/sdfnet_prepare_data.py` 将赛题数据转换为 `ship_reid_system/data/HOSS/{bounding_box_train,bounding_box_test,query}`：
- 复用现有 `local_val_task.json`（存在则不重新划分），`bounding_box_test`=gallery、`query`=query、`bounding_box_train`=labels_train 对应图像
- 文件命名遵循 SDF-Net `datasets/hoss.py` 约定（`{pid}_{cid}_{imgid}_RGB.tif / _SAR.tif`），SAR 图像保持**三通道灰度复制**原样（赛题硬约束，不改像素）
- 输出 `identity_map.json`（image_id ↔ pid ↔ HOSS 文件），供推理脚本回映射赛题 image_id

**环境**：Python 3.9+ / PyTorch 2.2.2+cu118 / timm==1.0.25 / yacs / opencv-python / Pillow（官方 requirements.txt）。`12_sdfnet_setup.bat` 自动创建 `ship_reid_system/.venv-sdfnet` 并安装上述依赖；16GB 显存机器建议 `SDF_IMS_PER_BATCH=32`。

**推理**：`12_sdfnet_zero_shot.bat` 用官方权重直接跑本地验证（`scripts/sdfnet_inference.py`，特征提取 + 按 query_type 分组后处理 + `evaluate.py --submission` 评测），**零训练成本**。

### 8.14 训练时间缩减策略

1. **零训练初判优先**：`13_sdfnet_plan.bat` 的 A1/A2 阶段用官方 HOSS 预训练权重直接本地验证（base / rerank+QE 两组合），不训练即可判断基线水平。
2. **短 epoch 微调（仅一次）**：A 阶段后入口自动检查微调权重 `logs/SDF-Net-finetune/best.pth`；不存在且本机有 CUDA 时自动微调（默认 **60 epoch**，`SDF_EPOCHS` 可配），以官方权重为起点续微调（官方默认 100 epoch，缩至 60 可显著控制训练时间）；B1/B2/D1~D4 共享该权重，不重复训练。
3. **决策节奏**：按 8.11 的原则逐项吃零成本增益；微调结果按本地验证 Δ（>0.005）判定是否保留，避免盲目加训。

### 8.15 rerank/QE 跨模态分组修复说明

**问题**：`scripts/transoss_inference.py` 的 `--rerank/--qe` 后处理原先对所有 query 在每个模态 gallery 子集统一执行，未区分 `query_type`，导致跨模态候选污染（如 O2O 查询混入 SAR 候选、O2S 查询混入光学候选），O2O 方向 R@1 严重偏低。

**修复**：后处理改为按 `query_type` 分组执行——
- O2S：只在 **SAR gallery** 上 rerank/QE；
- S2O：只在 **optical gallery** 上 rerank/QE；
- O2O：只在 **optical gallery** 上 rerank/QE，且额外排除 query 自身。

`scripts/sdfnet_inference.py` 采用同一分组逻辑，保证两模型后处理行为一致。

**重测**：修复后需用现有 `local_val_task.json` 重测 rerankqe 组合（`11_transoss_ab.bat` 的 rerankqe 项 / `13_sdfnet_plan.bat` 的 A2、B2、C 组合），确认 **O2O 回到 0.94+** 且综合相对 base 有正增益，才在最终提交中启用 rerank/QE。

### 8.16 跨模型融合（SDF-Net × TransOSS）

**脚本**：`scripts/fuse_sdfnet_transoss.py`，支持两种融合（可配置权重）：
- **weighted（加权平均）**：两模型 sim 矩阵分别 min-max 归一化到 [0,1] 后加权求和（`--weight_a` 为 SDF-Net 权重，默认 0.6）；
- **rrf（Reciprocal Rank Fusion）**：`score = Σ 1/(60+rank)`（未进该 query 前 10 计 0）。

**流程**：SDF-Net 与 TransOSS 分别做 base 推理并 `--save_sim` 导出 `sim.pt + meta.json` → 融合脚本按 `image_path` 对齐 → 逐 query 按融合分降序取前 10 → 复用 `evaluate.py --submission` 评测，输出单模型与融合成绩对比报告。

**一键**：`13_sdfnet_plan.bat` 的 D1~D4 阶段自动执行（D1/D2/D3 = weighted，SDF 权重 0.5/0.6/0.7；D4 = RRF）。融合需两模型均已就绪（SDF-Net 微调权重 + Hoss-ReID/transformer_200.pth）。

### 8.17 合规披露清单（复现材料）

| 类别 | 内容 |
|---|---|
| 预训练权重 | SDF-Net：HuggingFace `Chenfree233/SDF-Net` → `SDF-Net.pth`；TransOSS：HuggingFace `Alioth2000/TransOSS` → `vit_b512_pre.pth` |
| 外部代码库 | SDF-Net：GitHub `cfrfree/SDF-Net`（commit `15cbb1a88742256c9a6754d4f29f956991366f93`）；TransOSS：GitHub `Alioth2000/Hoss-ReID` |
| 依赖环境 | SDF-Net：Python 3.9+ / PyTorch 2.2.2+cu118 / timm==1.0.25 / yacs / opencv-python / Pillow（`.venv-sdfnet`）；TransOSS 与自研：见第 2 节（CUDA 12.4） |
| 训练/推理命令 | 见 8.1~8.18 各节：`12_sdfnet_setup.bat`（SDF 环境准备）/ `13_sdfnet_plan.bat`（SDF × TransOSS 统一方案入口）/ `07/08/09/10/11` 系列 bat |
| 随机种子 | SDF-Net：`config/defaults.py` `SOLVER.SEED=1949`（可由 `--opts SOLVER.SEED` 覆盖）；自研：`config/default.yaml` seed |
| 数据与预处理 | 赛题官方训练数据；HOSS 转换仅目录重排 + 三通道灰度复制，**不改像素**；SAR 增强（`--preprocess`）属推理端可选处理 |
| 赛题合规 | 所有提交格式校验复用 `evaluate.py`（O2S/S2O/O2O 分方向 + 综合得分）；候选模态严格匹配 query_type；不使用 AIS/航迹等未公开辅助信息 |

### 8.18 统一方案入口（13_sdfnet_plan.bat）

**定位**：把零散的 12 系列对比脚本收敛为单入口一键运行——按阶段自动执行 9 个组合的推理/微调/评测，并自动汇总对比文档，避免手动逐个跑脚本比对。

**组合清单**（全部复用固定 `local_val_task.json`，不重新划分；微调仅执行一次，其余组合共享权重零训练推理）：

| 组合 | 配置 | 说明 |
|---|---|---|
| A1 | SDF-Net 官方权重 base | 零训练基线 |
| A2 | SDF-Net 官方权重 + rerankqe | 官方权重推理端增强 |
| B1 | SDF-Net 微调后 base | 微调权重（默认 60 epoch，`SDF_EPOCHS` 可配） |
| B2 | SDF-Net 微调后 + rerankqe | 微调权重推理端增强 |
| C | TransOSS 现有权重 + rerankqe 重测 | 修复后分组逻辑重测 |
| D1~D3 | SDF(微调后) × TransOSS weighted | SDF 权重 0.5 / 0.6 / 0.7 |
| D4 | SDF(微调后) × TransOSS RRF | 倒数排名融合 |

**用法**：

```bat
cd /d D:\Projects\Multi-source Satellite Cross-modal Ship Re-Identification\ship_reid_system
13_sdfnet_plan.bat                    :: 默认：微调 60 epoch
set SDF_EPOCHS=100 && 13_sdfnet_plan.bat  :: 自定义微调 epoch（SDF_IMS_PER_BATCH / SDF_WEIGHT / TRANS_WEIGHT 同理）
```

**自动产出**：
- 对比文档：`experiments/exp_005_sdfnet_plan_compare.md`（完整结果表：各组合 O2S/S2O/O2O 的 R@1、mAP、综合分；每组合 Δ vs TransOSS baseline 0.4831/0.4881；正增益标记 >0.005；推荐提交组合与决策建议）
- 中间产物：`ship_reid_system/sims/plan/`（各组合 `pred_*.json`、`sim.pt + meta.json`、`run.log` 执行日志）

**失败/跳过策略**：微调权重不存在且无 CUDA 时 B/D 自动跳过并记录原因；TransOSS 侧环境缺失时 C/D 自动跳过；单个组合失败不影响其余组合继续执行；全部记录见 `sims/plan/run.log`。

**收敛说明**：`12_sdfnet_setup.bat`（一次性环境准备）保留；原 `12_sdfnet_zero_shot / 12_sdfnet_finetune / 12_sdfnet_fusion` 三个脚本已整合进本入口。

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
| TransOSS 训练在 EVAL_PERIOD 阶段报空张量错误 | 先运行 `08` 脚本创建 `query/`、`bounding_box_test/` 占位目录 |
| TransOSS 训练内置评估 mAP 虚高 | 评估用占位数据、身份与训练集重叠，数字不可信，以测试集推理 + 本地验证为准 |

