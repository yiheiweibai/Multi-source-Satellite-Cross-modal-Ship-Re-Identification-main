# 赛题06 多源卫星跨模态舰船重识别 · 代码系统框架

面向 **2026 全国大数据与计算智能挑战赛 赛题06「多源卫星跨模态舰船重识别」** 的可运行工程化代码框架。
技术方案参照同目录参赛技术方案文档：《2026全国大数据与计算智能挑战赛赛题06多源卫星跨模态舰船重识别参赛技术方案.md》。

## 1. 项目结构

```
ship_reid_system/
├── config/
│   ├── __init__.py          # config 包导出
│   ├── config.py            # Config dataclass（全量参数 + yaml 加载/保存）
│   └── default.yaml         # 默认配置（命令行 --config 可覆盖）
├── data/
│   ├── __init__.py
│   ├── dataset.py           # 光学/SAR 数据集接口（标注 json -> Dataset）
│   ├── sampler.py           # PK 采样器（P 身份 × K 图，优先跨模态身份）
│   ├── transforms.py        # 数据增强 + SAR 斑点滤波占位 + 斑点模拟增强
│   └── dummy.py             # 模拟数据生成器（数据开放前跑通全流程）
├── models/
│   ├── __init__.py
│   ├── backbone.py          # 双分支骨干：ResNet50 / ViT（浅层特定 + 深层共享）
│   ├── heads.py             # BNNeck 模态不变特征头
│   └── model.py             # ShipReIDModel 组装
├── losses/
│   ├── __init__.py
│   ├── supcon_loss.py       # 跨模态监督对比损失（SupCon）
│   ├── triplet_loss.py      # 三元组损失（含难样本挖掘/自适应 margin）
│   └── composed_loss.py     # SupCon + 三元组 + 身份分类 加权组合
├── trainers/
│   ├── __init__.py
│   ├── trainer.py           # 训练/验证循环、ckpt 保存恢复、tensorboard
│   └── lr_scheduler.py      # 分层学习率（骨干 0.1×）+ 调度器
├── inference/
│   ├── __init__.py
│   └── retrieval.py         # 特征提取 + 余弦检索 + Top-K 输出
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # mAP / Recall@K（整体 + 同模态 + 跨模态）
│   └── logger.py            # 日志封装
├── data/                    # 数据目录（运行时自动创建）
│   ├── annotations/         # 标注 json：train.json / val.json
│   └── dummy/               # 模拟图像（optical/ 与 sar/）
├── outputs/                 # 训练产物（自动创建）
│   ├── logs/                # tensorboard
│   ├── checkpoints/         # last.pth / best.pth
│   └── retrieval_results.json
├── requirements.txt
├── README.md
├── train.py                 # 训练入口
├── inference.py             # 推理/检索入口
└── evaluate.py              # 评测入口
```

## 2. 环境安装

**推荐（Windows GPU 一键安装，RTX 4070 Ti SUPER / CUDA 12.4）**：双击根目录 `setup.bat`，
或在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
```

脚本会自动：创建 `.venv` 虚拟环境 → 升级 pip → 以 CUDA 12.4 wheel 索引安装
torch/torchvision（兼容 Ada 架构）→ 安装 `requirements.txt` 其余依赖 → 打印
Python / torch / `cuda.is_available()` / GPU 名称与显存。脚本幂等可重跑，失败时
退出码非 0 并输出明确错误信息。详见第 7 节。

**手动安装**：

```bash
# 1) 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 2) CUDA 12.4 版 PyTorch（RTX 40 系 Ada 架构）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3) 其余依赖（torch/torchvision 行已注释，勿用默认 PyPI 覆盖）
pip install -r requirements.txt
# 可选：真实 SAR 斑点滤波（Lee/Kuan/BM3D）依赖已含 opencv-python scipy
```

- 纯 CPU 兜底（无 NVIDIA 驱动时）：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- 建议驱动 ≥ 551.x（CUDA 12.4 对应），可在 NVIDIA 官网更新。

## 3. 数据放置约定

**标注格式（JSON）**：列表，每项一个样本：

```json
[
  {"image_path": "data/raw/optical/ship_001.jpg", "identity": 0, "modality": "optical"},
  {"image_path": "data/raw/sar/ship_001.png",     "identity": 0, "modality": "sar"}
]
```

- `identity`：整数身份 ID，同一艘船 = 同一 ID
- `modality`：`optical` / `sar`；缺省时按路径关键字自动推断
- 数据开放后：把 `config/default.yaml` 中 `dummy_mode` 置 `false`，并将
  `train_ann` / `val_ann` 指向官方标注（或按上述格式自建标注），填好路径即可训练。

**模拟数据**：`dummy_mode: true` 时若标注缺失，自动生成 40 个身份 × 光学/SAR
的模拟图像到 `data/dummy/` 并写出标注 `data/annotations/{train,val}.json`，可直接跑通全流程。

## 4. 使用流程

### 4.1 训练（默认模拟数据）

```bash
# 用默认配置（dummy 数据）训练
python train.py

# 自定义配置 + 覆盖训练轮数（可复制 config/default.yaml 再修改）
python train.py --config config/default.yaml --epochs 60

# 断点续训
python train.py --resume outputs/checkpoints/last.pth
```

- 训练日志：stdout + `outputs/logs/`（tensorboard）
- checkpoint：`outputs/checkpoints/last.pth`（最新）、`outputs/checkpoints/best.pth`（验证 mAP 最优）
- 每 epoch 结束自动在验证集评测 mAP / Recall\@1/5/10

### 4.2 推理（检索）

```bash
python inference.py --ckpt outputs/checkpoints/best.pth --topk 10
# 指定查询集/图库（数据开放后）
python inference.py --ckpt outputs/checkpoints/best.pth --query_ann data/annotations/query.json --gallery_ann data/annotations/gallery.json
```

- 输出：`outputs/retrieval_results.json`（每个查询的 Top-K 候选路径）

### 4.3 评测

```bash
python evaluate.py --ckpt outputs/checkpoints/best.pth
python evaluate.py --ckpt outputs/checkpoints/best.pth --ann data/annotations/val.json --k 1 5 10
```

- 输出：整体 mAP / Recall\@K，以及同模态（same\_*）与跨模态（cross\_*）子集指标

## 5. 关键设计（对应技术方案）

| 模块  | 设计                                                                                         |
| --- | ------------------------------------------------------------------------------------------ |
| 骨干  | 双分支：浅层模态特定（光学/SAR 各一份，权重独立），深层共享（ResNet `layer3` 起 / ViT 前 6 个 block 特定）。可用 ImageNet 预训练权重 |
| 特征头 | BNNeck：BN 前特征 L2 归一化用于检索，BN 后特征接分类头，解耦度量与分类                                                |
| 损失  | 跨模态 SupCon（身份为正样本对）+ 难样本三元组（可自适应 margin）+ 身份分类（标签平滑），按权重加权                                 |
| 训练  | PK 采样（优先双模态身份）、分层学习率（骨干 0.1×）、梯度裁剪、cosine/OneCycle 调度、ckpt 恢复、tensorboard                  |
| 推理  | 余弦相似度 Top-K 排序，输出 json                                                                     |
| 评测  | mAP / Recall\@K，整体 + 同模态 + 跨模态子集                                                           |
| 数据  | SAR 斑点滤波占位（恒等），训练时对 SAR 图模拟斑点噪声增强；数据开放后替换真实滤波                                              |

## 6. 真实数据接入（已适配官方格式）

### 6.1 训练：官方 labels.csv 直接可用

`data/dataset.py` 提供 `ShipReIDCSVDataset` / `load_labels_csv`，直接解析官方
`labels.csv`（四列：`image_id,ship_id,modality,image_path`），图像自动定位到数据包根目录。

```yaml
# 示例：自定义 yaml（dummy_mode 必须为 false）
dummy_mode: false
data_root: "../赛题6-初赛/训练数据"   # 相对于 ship_reid_system/ 目录
train_labels_csv: "../赛题6-初赛/训练数据/labels.csv"   # 或 train_labels_csv: "<data_root>/labels.csv"
val_labels_csv: ""            # 不设验证集则每个 epoch 仍保存 best.pth
auto_num_classes: true        # 自动读取 ship_id 数
```

```bash
python train.py --config my_train.yaml
```

### 6.2 推理：生成赛题提交 prediction.json

`inference.py --task_json` 模式解析官方测试包 `task.json`，按 query\_type 过滤候选模态
（O2S→sar、S2O→optical、O2O→optical），对每个 query 取相似度降序前 10 个不重复
gallery image\_id，生成并保存 `prediction.json`（UTF-8、顶层 JSON 对象、仅含
`query_id -> [10 个 image_id]`），保存前自动格式自检（全覆盖/恰好10个/不重复/ID合法/模态匹配）。

```bash
python inference.py --config my_train.yaml --ckpt outputs/checkpoints/best.pth \
    --task_json "../赛题6-初赛/初赛测试数据/task.json" \
    --out_prediction outputs/prediction.json
```

冒烟调试参数（正式提交勿用）：`--max_queries_per_dir N`（每方向最多 N 条 query）、
`--max_gallery_per_mod M`（每模态最多 M 个 gallery）。

### 6.3 评测：赛题方向指标

`evaluate.py --submission` 对齐赛题评测：对 O2S/S2O/O2O 分别计算 R\@1 与 mAP\@10，
方向得分=两者平均，综合得分=0.45×O2S+0.45×S2O+0.10×O2O。

```bash
python evaluate.py --config my_train.yaml --submission \
    --task "../赛题6-初赛/初赛测试数据/task.json" \
    --prediction outputs/prediction.json \
    --gt ground_truth.json     # 组织方保留的 GT；本地可用自建模拟 GT 冒烟
```

## 7. GPU 正式训练（RTX 4070 Ti SUPER / CUDA 12.4）

### 7.1 一键安装环境

双击 `setup.bat`，或在项目根目录 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
```

脚本行为：

1. `python -m venv .venv` 创建虚拟环境（已存在则复用，幂等）；
2. 激活环境并升级 pip；
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
   安装 CUDA 12.4 版 PyTorch（适配 Ada 架构，如 RTX 4070 Ti SUPER）；
4. `pip install -r requirements.txt` 安装其余依赖（opencv-python/scipy 已启用）；
5. 自动验证并打印：Python 版本、torch 版本、`cuda.is_available()`、
   GPU 名称与显存（`torch.cuda.get_device_name` / `get_device_properties`）；
6. 任一步失败：打印 `[ERROR] ...` 明确原因并退出码非 0（`setup.bat` 会停留报错）；
7. 重复运行安全：`.venv` 复用、已装 cu124 torch 跳过。

### 7.2 启动 GPU 训练

```bash
# 激活环境（setup.bat 后无需重复）
.\.venv\Scripts\activate

# 使用 GPU 正式训练配置（epochs=120, num_workers=8, pk_p=8 pk_k=4,
# lr=3e-4, backbone_lr_scale=0.1, resume 空）
python train.py --config config/train_gpu.yaml
```

- 数据路径已在 `config/default.yaml` 指向真实训练数据 `labels.csv` 与初赛 `task.json`，
  `dummy_mode: false` 完全禁用模拟数据；
- `val_labels_csv` 为空 → 每 epoch 保存 `best.pth`（按训练集度量）；
- checkpoint：`outputs/checkpoints/{last,best}.pth`，断点续训填 `resume`；
- 推理生成提交：见 6.2 节 `inference.py --task_json`。

### 7.3 常见问题（FAQ）

| 现象                                   | 排查                                                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `cuda.is_available()` 为 False        | ① NVIDIA 驱动过旧：建议 ≥ 551.x（对应 CUDA 12.4），到 NVIDIA 官网更新后重启；② torch 装成了 CPU 版：卸载后按 2 节 cu124 索引重装；③ 执行环境未激活 `.venv` |
| `setup_env.ps1` 报 `Python not found` | 安装 Python 3.8+ 并勾选 "Add Python to PATH"，重开终端再跑                                                                  |
| 安装 torch 网络失败/超时                     | 更换镜像或重跑脚本（幂等，不会重复下载已成功部分）；确认可访问 `download.pytorch.org`                                                          |
| 显存不足（OOM）                            | 降低 `pk_p`/`pk_k`（如 8×4→6×4）或 `batch` 等效值；或关闭 `sar_speckle_aug` 降内存；12GB 显存下 resnet50+32 batch 通常可跑              |
| 训练慢/卡死                               | `num_workers` 8 在 32GB 内存下足够；若 CPU 瓶颈可降为 4；确认无其他进程占用 GPU（`nvidia-smi`）                                          |
| 驱动能跑 CUDA 但版本提示 mismatch             | torch cu124 需要驱动支持 CUDA 12.x 运行时；驱动更新至 ≥ 551 即可，无需安装完整 CUDA Toolkit                                             |
| AMP 混合精度                             | 框架当前未实现 `autocast`/`GradScaler`，`train_gpu.yaml` 未开启；后续支持后可增加 `amp: true`                                       |

