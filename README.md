---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: a88da0184087caa3b0b1fbad5b044954_05bd5e49c55911f197b3525400248c00
    ReservedCode1: rF49NKMAPtZWjrwMjaWURtMvSPqYm49VyGAoYEFr8BTSvObrBvJvPHMk8VM6aseJphpFD3s7Tr2MNvSinxdUEDh2X0/FTRhJrZyTSajfNv14poG/NKM8HRjuXAiUk5IHZYO/fw5N07lMfh9eEsv1RSwNKgnrlQGwsaZ2l9OM+qCaHMOD7NngFSyBfbc=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: a88da0184087caa3b0b1fbad5b044954_05bd5e49c55911f197b3525400248c00
    ReservedCode2: rF49NKMAPtZWjrwMjaWURtMvSPqYm49VyGAoYEFr8BTSvObrBvJvPHMk8VM6aseJphpFD3s7Tr2MNvSinxdUEDh2X0/FTRhJrZyTSajfNv14poG/NKM8HRjuXAiUk5IHZYO/fw5N07lMfh9eEsv1RSwNKgnrlQGwsaZ2l9OM+qCaHMOD7NngFSyBfbc=
---

# 赛题06 多源卫星跨模态舰船重识别 · 复现指南

> **一句话**：光学（Optical）与 SAR 两种模态下同一艘舰船的跨模态身份匹配。最终方案 = **11 成员 RRF 倒数排名融合**（3 个 SDF-Net 全量 checkpoint + 7 个异构分支成员：ViT / ConvNeXt-Tiny / 强配方 ConvNeXt-T / ConvNeXt-Small 强配方 / Swin-Small 强配方 / DINOv2-B ×2 + 队友 `sdfnet_base75` 互补成员），`member_topk=100`、`k=10`、每 query 输出 top-10。
> **当前 active 预设 = `ensemble.active: final_full_b75_cm_dino_nocnxs_dir`**（波次10，2026-10-12）：波次5 全量重训（官方 `labels.csv` 3534 身份 / 7422 图）为基座 → 波次9 剪去最弱成员 `ship_reid_convnext_s` → 波次10 叠加 O2S 方向覆盖（`sdfnet_mos75_full=1.5 / ship_reid_vit_full=0.5 / ship_reid_convnext_t_full=0.25`）。
> **冠军提交链 = 该 11 成员 RRF 基座 → S5 互反重排（`scripts/s5_rebuild.py`）→ 闭配 α 簇修正（`scripts/pair_match.py` + `pair_verify.py` + `pair_anchors.json`）**，**官方 Public = 0.862**（α 版，即根目录 `prediction.json`）。因 11 个成员全为全量重训（已见全部 3534 身份），融合整体**无无泄漏本地验证**（盲提交）；缩水口径下的无泄漏消融（波次 8~10）是唯一的量化证据。
> **官方分演进**：波次6 **0.763125** → 波次11 0.758 → 波次12 0.741（判负、已回滚）→ S5 基座 → 闭配 α **0.862**。**回滚点**：`prediction_s5.json`（S5 基座）与 `prediction_ag_bak.json`（α+γ 版 0.856）。
> **2026-10-18 「冠军链精简」清理**：主工程 91.9 GB → 18.4 GB，仅保留上述冠军链可重跑所需资产；2828 缩水时代的 checkpoint / log / 稠密 sim / 旧代码树已删除（详见 §2 末注），`reproduce.yaml` 的历史预设因此多数**不再可重跑**。

本文档面向「能在本机把最终提交复现出来」的人。技术选型理由见 `技术方案.md`，历史取舍与后续方向见 `路线方案.md`。

---

## 1. 当前状态

| 项 | 值 |
|---|---|
| 赛题 | 2026 全国大数据与计算智能挑战赛 · 赛题06「多源卫星跨模态舰船重识别」 |
| 测试集 | 267 query / 1800 gallery（O2S 120 / S2O 120 / O2O 27） |
| 选模判据 | `Final = 0.45×O2S + 0.45×S2O + 0.10×O2O`（方向分 = (R@1 + mAP@10)/2） |
| 冠军方案 | **11 成员 RRF 基座（`configs/reproduce.yaml` 的 `ensemble.active: final_full_b75_cm_dino_nocnxs_dir`，k=10 / member_topk=100 / topk=10 / O2S 方向覆盖）+ S5 互反重排 + 闭配 α 簇修正** |
| 官方成绩 | **Public 0.862**（闭配 α 版，2026-10-13 提交的 `prediction.json`）；历史 0.763125（波次6）/ 0.758（波次11）/ 0.741（波次12，判负回滚） |
| 本地验证 | 冠军预设 **N/A（盲提交：11 个全量成员已见全部 3534 身份，不存在无泄漏验证）**；唯一的量化证据是「缩水成员同构子集」的无泄漏消融（波次8 12 成员 k=10 local_val 0.7025 → 波次9 剪枝 +0.0066 → 波次10 O2S 覆盖 +0.0042），以及 `sdfnet_base75` 单成员本地综合 0.5985 |
| 单模型冠军基线 | E0 = SDF-Net ep80（缩水版）：0.5888；MOS ep75（缩水版）：0.6118 |
| 最终提交文件 | 工程根上级 `prediction.json`（`out_prediction: ../prediction.json`） |
| 赛题数据路径 | 工程根上级 `question6-data\`：`preliminary-round-test-data\task.json`（test_task）、`traindata\local_val_task.json` + `local_val_gt.json`（local_val）。**工程内没有 `赛题6-初赛\`** |
| 工程体积 | 清理后 ~18.4 GB（清理前 ~91.9 GB） |
| 项目 Git | 工程根目录含 `.git`（历史提交只读，无指令不做新提交）；`logs\`、`outputs\`、`*.pth`、`*.pt`、`venv\` 均在 `.gitignore` 内，删除即不可恢复 |

---

## 2. 目录结构（与磁盘一致）

工程根目录 = `h:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\`（已扁平化，**没有** `ship_reid_system\` 这一层）。

```
h:\Ship-Re-Identification\
├── question6-data\                     赛题数据（reproduce.yaml 实际引用）
│   ├── traindata\                      labels.csv / labels_train.csv / local_val_task.json / local_val_gt.json
│   └── preliminary-round-test-data\    task.json（初赛测试集）
├── prediction.json                     ★ 冠军提交文件（267 query, top-10；α 版，Public 0.862）
├── prediction_s5.json                  S5 基座（回滚点）
├── prediction_ag_bak.json              α+γ 版（0.856）备份
├── prediction_probe_alpha.json         α 版源文件
├── prediction_closedpair.json / pair_reemit.json   闭配 α 产物（同一内容）
├── prediction_probe_wcons.json / prediction_probe_wbase.json   求解探针（非提交）
└── Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\   ← 工程根目录
    ├── 12_sdfnet_setup.bat             环境准备（venv / 依赖 / 官方权重 / HOSS 数据）
    ├── 13_train_champion.bat           重训冠军链的 8 个 full 成员（5 个 ship_reid 全量配置 + 2 个 SDF-Net 全量配置；逐成员断点跳过）
    ├── 14_submit_1_env_check.bat       ① 环境自检
    ├── 15_submit_2_sdfnet_infer.bat    ② SDF-Net 成员推理（当前预设 3 个全量成员）
    ├── 16_submit_3_shipvit_infer.bat   ③ ship_reid 系成员推理（当前预设 7 个）
    ├── 17_submit_4_rrf_fuse.bat        ④ RRF 融合
    ├── 18_submit_5_verify.bat          ⑤ 提交格式终检
    ├── configs\
    │   └── reproduce.yaml              ★ 唯一事实来源（路径 / 成员池 / 权重预设 / k / 方向覆盖）
    ├── config\                         自研 ship_reid 系旧配置（train_gpu.yaml 等）
    ├── scripts\
    │   ├── repro_cfg.py                解析 reproduce.yaml（相对路径均相对工程根目录）
    │   ├── rrf_fuse.py                 RRF 融合 + 平台级格式校验
    │   ├── sdfnet_inference.py         单模型推理 + 按 query_type 分组后处理
    │   ├── s5_rebuild.py               ★ S5 互反重排（--fidelity 可 267/267 复现部署物）
    │   ├── pair_match.py / pair_verify.py / pair_anchors.json   ★ 闭配 α 簇修正
    │   ├── sdfnet_ckpt_sweep.py        checkpoint 选优（按 Final 排序）
    │   ├── sdfnet_prepare_data.py      赛题数据 → HOSS 布局
    │   ├── sdfnet_prepare_full.py      全量数据 → HOSS 布局（波次5，data/full）
    │   ├── sdfnet_download_weights.py  官方权重下载
    │   ├── check_prediction.py         提交格式独立校验器
    │   └── build_local_val.py / compute_stats.py / b1_topk_probe.py / member_screen.py
    ├── data\  models\  inference\  utils\   自研模块（保留）
    ├── evaluate.py                     赛题提交评测入口（--submission）
    ├── SDF-Net\                        主骨干（上游 cfrfree/SDF-Net + 本项目改动）
    │   ├── configs\SDF-Net.yml         推理用配置（256×256）
    │   ├── configs\SDF-Net-full.yml    波次5 全量重训配置（唯一差异：DATASETS.ROOT_DIR=data/full）
    │   ├── configs\SDF-Net-mos-full.yml  波次5 全量 MOS 配置（CMAL_LOSS_WEIGHT=1.0）
    │   ├── loss\cmal_loss.py           CMAL 对齐损失
    │   └── ...
    ├── ship_reid_vit\                  异构分支（与 SDF-Net 并列）
    │   ├── 01_setup.bat … 05_inference.bat      （04 / 05 为 legacy 2828 入口，不在冠军链内）
    │   ├── config\train_*_full.yaml              现役 5 个全量成员的推理/训练配置
    │   ├── config\train_convnext_t_strong_cm_full.yaml   强配方 ConvNeXt-T 全量配置
    │   ├── config\train_dinov2_b.yaml            DINOv2-B 缩水版配置（现役成员）
    │   ├── outputs\full_vit\checkpoints\epoch_100.pth                 ★ ship_reid_vit_full 权重
    │   ├── outputs\full_convnext_t\checkpoints\epoch_080.pth          ★ ship_reid_convnext_t_full 权重
    │   ├── outputs\full_convnext_t_strong_cm\checkpoints\epoch_080.pth ★ ship_reid_convnext_t_strong_cm_full 权重
    │   ├── outputs\full_convnext_s_pure\checkpoints\epoch_040.pth     ★ ship_reid_convnext_s_pure_full 权重
    │   ├── outputs\full_swin_s_v2\checkpoints\epoch_060.pth           ★ ship_reid_swin_s_v2_full 权重
    │   ├── outputs\screen_dinov2_b\checkpoints\epoch_040.pth / epoch_060.pth  ★ 两个 DINOv2-B 成员权重
    │   ├── venv\                       16 号推理环境
    │   └── weights\vit_base_patch16_224.pth
    ├── logs\
    │   ├── SDF-Net\                    官方权重 `SDF-Net.pth` + 256×256 衍生版 `SDF-Net_256.pth`
    │   ├── SDF-Net-full\               波次5 全量（transformer_25/45.pth） ★ 成员 sdfnet_ep25/45_full 来源
    │   └── SDF-Net-mos-full\           波次5 全量 MOS（transformer_75.pth） ★ 成员 sdfnet_mos75_full 来源
    ├── sims\
    │   ├── test\top100_*.json              缩水/dino/队友成员的测试集排名（含 `top100_base75.json`）
    │   ├── test_full\top100_*_full.json    全量成员测试集排名（15/16 号产出）
    │   ├── test_rerank\sdfsim_{mos75,ep45,ep25}_full\  S5 所需 3 个 SDF-Net 稠密 sim
    │   └── test_rerank_full\               S5 所需 7 个 ship_reid 全量稠密 sim（`sim_*.pt`）
    ├── experiments\                    实验记录（exp_001~007、exp_010）
    └── *.official.md                   赛题官方起始模板（历史存档，已过期，见 §7）
```

> **2026-10-18 清理删除了**（不可恢复，均在 `.gitignore` 内）：`logs\SDF-Net-finetune\`、`logs\SDF-Net-mos\`、`ship_reid_vit\outputs\screen_convnext_*\`、`ship_reid_vit\outputs\checkpoints\`、`ship_reid_vit\outputs\full_convnext_s\`、`ship_reid_vit\outputs\full_dinov2_b\`、`sims\localval*\`、`sims\test_rerank_full_ms\`、`ship_reid_system\`（整树）、`question6-data - 副本\` 等。**直接后果**：`reproduce.yaml` 里的 `final`（2828 版 8 成员）、`final_full`、`final_full_b75`、`..._ms`、`..._dino_full_...` 等历史预设因 checkpoint 缺失而**无法重跑**；只有 active（`final_full_b75_cm_dino_nocnxs_dir`）及其 `per_direction` 变体可端到端重跑。

---

## 3. 环境准备

需要**两个**独立虚拟环境。

### 3.1 `.venv-sdfnet`（14~18 号脚本使用）— **已就绪**

`.venv-sdfnet` 当前已重建完成（Python 3.11，`torch 2.2.2+cu118` / `torchvision 0.17.2+cu118` / `timm 1.0.25` / `numpy 1.25.0`，CUDA 可用）。若需从零重建：

```bat
:: 在工程根目录双击或执行：
12_sdfnet_setup.bat
```

该脚本幂等地完成：
1. 创建 `.venv-sdfnet`（Python 3.9+）
2. 安装 `torch==2.2.2+cu118` / `torchvision==0.17.2+cu118` + `numpy/opencv-python/Pillow/thop/timm/yacs`（无 NVIDIA GPU 时脚本会提示改用 CPU 版）
3. 检查上游 `SDF-Net`（已存在且含 `train.py` 则跳过）
4. 从 HuggingFace `Chenfree233/SDF-Net` 下载官方权重 → `SDF-Net\logs\SDF-Net\SDF-Net.pth`（约 333MB）
5. 派生 256×256 版本 → `SDF-Net\logs\SDF-Net\SDF-Net_256.pth`：官方权重是在 256×128 下训练的（`pos_embed` 为 `[1,130,768]`），直接配 256×256 输入会形状不匹配，故调 `scripts\sdfnet_resize_posembed.py` 把 pos_embed 双线性插值到 `[1,258,768]`（插值期间 cls / spe 两个前缀 token 原样保留）
6. 把赛题数据转成 HOSS 布局 `SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}`，**复用现有 `local_val_task.json`，不重新划分**

可用 `SDF_DATA_DIR` 覆盖数据目录。12 号结束后直接进入 §4（从 14 号开始）。

> **若 `.venv-sdfnet` 存在但缺 `Scripts\`**（安装被截断）：删除整个 `.venv-sdfnet` 后重跑 `12_sdfnet_setup.bat`。本机 2026-10-18 曾出现该状态（`torch` 仅 31 MB、`Scripts\` 缺失），已修复。

> **两个权重文件的分工**：`SDF-Net.pth` 是官方原始 256×128 权重（下载产物）；**`SDF-Net_256.pth` 才是本项目 256×256 链路实际使用的那份**——`reproduce.yaml` 的 `sdfnet_official` 与 `SDF-Net.yml` / `SDF-Net-mos-full.yml` 的 `PRETRAIN_PATH`、`WEIGHT` 均指向它，14 号自检检查的也是它。两者都由 12 号自动准备完成，无需手工干预。

> **波次5 全量数据目录**：冠军链的 3 个 SDF-Net 成员用 `SDF-Net\data\full\HOSS\`（由 `scripts\sdfnet_prepare_full.py` 生成，训练集为全量 `labels.csv` 的 7422 图；query/gallery 以 junction 复用现役目录）。该目录与 `data\HOSS` 平级，不在 12 号默认流程内。

### 3.2 `ship_reid_vit\venv`（仅 16 号 ship_reid 系成员推理使用）

```bat
:: 进入 ship_reid_vit 目录后执行：
01_setup.bat
```

创建 `ship_reid_vit\venv`、安装 torch/torchvision（CUDA 12.4，失败回落 CPU）+ `requirements.txt`，并下载 `weights\vit_base_patch16_224.pth`。

> 注意：`ship_reid_vit\03_train.bat` 会生成 2828 缩水版权重 `outputs\checkpoints\best.pth`，**该权重已在 2026-10-18 清理中删除，且不在冠军链内**；现役成员权重全部来自 `outputs\full_*\` 与 `outputs\screen_dinov2_b\`（见 §2 目录树）。

---

## 4. bat 脚本（12、13、14~18）与提交流程

### 4.0 全部 bat 一览（12、13、14~18）

| # | 脚本 | 定位 | 做什么 | 运行环境 | 主要输入 → 输出 |
|---|---|---|---|---|---|
| 12 | `12_sdfnet_setup.bat` | 环境准备（一次性） | 建 `.venv-sdfnet`；装 `torch 2.2.2+cu118` / `torchvision 0.17.2+cu118` + `numpy/opencv-python/Pillow/thop/timm/yacs`；检查上游 SDF-Net；从 HuggingFace `Chenfree233/SDF-Net` 下载官方权重 `SDF-Net.pth`（约 333MB）；**派生 256×256 版本 `SDF-Net_256.pth`**（pos_embed 插值，链路真正使用的那份）；把赛题数据转成 HOSS 布局 | 系统 Python（脚本自建 venv） | `question6-data\traindata\` → `.venv-sdfnet\`、`SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}`、`SDF-Net\logs\SDF-Net\SDF-Net.pth` + `SDF-Net_256.pth` |
| 13 | `13_train_champion.bat` | 训练（一次性，可断点续跑） | 重训冠军链的 **8 个 full 成员**：5 个 ship_reid 全量配置（`train_{vit,convnext_t,convnext_t_strong_cm,convnext_s_pure,swin_s_v2}_full.yaml`）+ 2 个 SDF-Net 全量配置（`SDF-Net-full.yml` → ep25/ep45，`SDF-Net-mos-full.yml` → ep75）。逐成员按「冠军 checkpoint 是否已存在」跳过；续训检查 `outputs\<dir>\checkpoints\last.pth`；SDF-Net 全量训练数据 `data\full\HOSS` 缺失时自动跑 `sdfnet_prepare_full.py` 生成。**不覆盖**：`sdfnet_base75`（队友 D 盘产物，本机仅有排名）、`ship_reid_dinov2_b_ep040/_ep060`（来自 `screen_dinov2_b`，未在本脚本内重训） | 两个 venv：`ship_reid_vit\venv`（5 个 ship_reid）+ `.venv-sdfnet`（2 个 SDF-Net） | `labels.csv` + 5 个 `train_*_full.yaml` + `SDF-Net-{full,mos-full}.yml` → `ship_reid_vit\outputs\full_*\checkpoints\epoch_*.pth`、`logs\SDF-Net-{full,mos-full}\transformer_*.pth` |
| 14 | `14_submit_1_env_check.bat` | **提交链 ①** | 解析 `reproduce.yaml` 把路径导出为环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项自检当前预设所需输入是否齐备 | `.venv-sdfnet` | `configs\reproduce.yaml` → 控制台 OK/MISS 清单；齐备退出码 0 |
| 15 | `15_submit_2_sdfnet_infer.bat` | **提交链 ②** | 对当前预设中**带 `ckpt` 的 SDF-Net 成员**逐个在测试集推理（cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）；每成员输出 `--topk %RRF_MEMBER_TOPK%` 个候选（当前 100）。**新增守卫：某成员 ckpt 不在本机但其 `pred` 已存在时跳过（复用现有预测）**；当前预设 `sdfnet_base75` 的 `ckpt` 已置空，不进推理步骤，不再阻断本机链路 | `.venv-sdfnet` | 3 个全量 SDF-Net ckpt（`logs\SDF-Net-mos-full\transformer_75.pth`、`logs\SDF-Net-full\transformer_45/25.pth`）→ `sims\test_full\top100_{mos75,ep45,ep25}_full.json`；`sdfnet_base75` 的 `pred`（`sims\test\top100_base75.json`）由 17 号直接读取 |
| 16 | `16_submit_3_shipvit_infer.bat` | **提交链 ③** | 遍历当前预设中**声明了 `vit_ckpt` 的 ship_reid 系成员**（当前 7 个：`ship_reid_vit_full` / `convnext_t_full` / `convnext_t_strong_cm_full` / `convnext_s_pure_full` / `swin_s_v2_full` / `dinov2_b_ep040` / `dinov2_b_ep060`），逐个用 `inference.py` 推理（`--topk %RRF_MEMBER_TOPK%`）。**默认逐成员复用：该成员的 `pred` 已存在则跳过、不重跑**（保证结果可比）；要重跑加参数 `force`。配置与权重来自成员自身的 `vit_config` / `vit_ckpt`（相对 `ship_reid_vit`），不是硬编码 | `ship_reid_vit\venv` | 7 组 config + ckpt（`train_*_full.yaml` + `outputs\full_*\checkpoints\epoch_*.pth`；dino 用 `train_dinov2_b.yaml` + `outputs\screen_dinov2_b\checkpoints\epoch_*.pth`）→ `sims\test_full\top100_*_full.json`（dino 为 `..._v2.json`） |
| 17 | `17_submit_4_rrf_fuse.bat` | **提交链 ④** | RRF 倒数排名融合：`score(c) = Σ_m w_m / (k + rank_m(c))`，`k=10`、每成员暴露 top-100 排名、每 query 取 top-10，并对 O2S 应用 `per_direction` 方向覆盖；成员与权重在 Python 侧按序绑定（不经过 cmd） | `.venv-sdfnet` | 当前预设 11 个成员预测文件（8 个来自 `sims\test_full\`、`sims\test\top100_base75.json`、dino 两个 `_v2.json`） → 工程根上级 `prediction.json`（工作副本 `submission_rrf.json`） |
| 18 | `18_submit_5_verify.bat` | **提交链 ⑤** | 调 `scripts\check_prediction.py` 逐条对照平台「直接判定无效」条件核验提交文件，并回显本地验证基准 | `.venv-sdfnet` | `prediction.json` + `task.json` → PASS/FAIL 报告 |

> 分工：**14→15→16→17→18** 是唯一的提交链路（下文 §4.1~§4.5 逐步展开）；**12** 是一次性环境准备；**13** 是一次性权重重训（仅在需要重建冠军成员权重时跑，已有 ckpt 会逐个跳过）。**端到端从零到提交的完整顺序见 §4.6。**
> 冠军链在这条 RRF 链之后还叠了两层**独立的**后处理：S5 互反重排（`scripts\s5_rebuild.py`）与闭配 α 簇修正（`scripts\pair_match.py` + `pair_verify.py`），它们**不在 14~18 号链路内**，按需手动执行（见 §4.7）。

统一约定：所有步骤在**工程根目录**执行；每一步都会重新解析 `configs/reproduce.yaml`（不读取陈旧的环境变量文件）。RRF 产物链路（当前 active 预设）：

```
成员预测（SDF-Net 侧 sims/test_full/top100_{mos75,ep45,ep25}_full.json
          + ship_reid 侧 sims/test_full/top100_shipvit_full.json / top100_convnext_t_ep080_full.json
                       / top100_convnext_t_strong_cm_full.json / top100_convnext_s_pure_ep040_full.json
                       / top100_swin_s_v2_ep060_full.json / top100_dinov2_b_ep040_v2.json / top100_dinov2_b_ep060_v2.json
          + 队友 base75 侧 sims/test/top100_base75.json）   ← 共 11 个成员
        └────► 17 号 RRF 融合（k=10, O2S 方向覆盖）──► ../prediction.json（RRF 基座）
                              └──► s5_rebuild.py（S5 = RRF_base + 0.5·R2）──► prediction_s5.json
                                        └──► pair_match.py + pair_verify.py（闭配 α）──► prediction.json（Public 0.862）
```

### 4.1 `14_submit_1_env_check.bat` — 环境自检

- 做什么：解析 `reproduce.yaml` → 导出环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项检查当前预设（`final_full_b75_cm_dino_nocnxs_dir`）所需输入是否齐备。
- 输入：`.venv-sdfnet`、`SDF-Net.yml`、官方权重 `SDF-Net_256.pth`、赛题 `task.json`；带 `ckpt` 的 3 个全量 SDF-Net 成员；ship_reid 系 7 个成员的 venv 与各自 `vit_ckpt`（`outputs\full_*\checkpoints\epoch_*.pth` 与 `outputs\screen_dinov2_b\checkpoints\epoch_*.pth`）。
- 输出：控制台 OK/MISS 清单；全部必需项齐备时退出码 0。**成员预测文件不算必需项**（由 15/16 号本地推理产出），缺失只显示为警告，不阻断。
- **本机现状（2026-10-18）**：**28 项全 OK，0 MISS**（退出码 0）。`sdfnet_base75` 的 `ckpt` 已置空（其权重在队友 D 盘 `D:/Projects/.../ship_reid_system/outputs/ab75/base75_best.pth`，本机不存在），自检改由其 `pred` 存在性把关，15 号也不再尝试推理该成员。
- 失败怎么办：按 `[MISS]` 行补齐——venv 缺失回 §3；ckpt/official 缺失跑 `12_sdfnet_setup.bat`；ship_reid 侧缺失跑 `ship_reid_vit\01_setup.bat` 或备齐对应 `vit_ckpt`。
- 可用 `--preset <名>` 临时切换预设做自检（但多数历史预设的 ckpt 已在清理中删除，会大量 MISS）。

### 4.2 `15_submit_2_sdfnet_infer.bat` — SDF-Net 成员推理

- 做什么：对当前预设中**带 `ckpt` 的成员**逐个在测试集推理（`scripts\sdfnet_inference.py`，`--batch_size 16`，`--topk %RRF_MEMBER_TOPK%`，cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）。**若某成员 ckpt 不在本机而其 `pred` 已存在，则打印 skip 并复用现有预测**；两者都不存在才报错退出。
- 输入：`logs\SDF-Net-mos-full\transformer_75.pth`（锚点）、`logs\SDF-Net-full\transformer_45.pth`、`transformer_25.pth`；`question6-data\preliminary-round-test-data\task.json`。（`sdfnet_base75` 的 `ckpt` 已置空，不进推理步骤，仅由 17 号直接读取其 `pred` = `sims\test\top100_base75.json`。）
- 输出：`sims\test_full\top100_mos75_full.json` / `top100_ep45_full.json` / `top100_ep25_full.json`（`SDF_N=3`）。
- 失败怎么办：检查 ckpt 路径（3 个 `*_full` 成员在 `reproduce.yaml` 里写绝对路径）、显存（OOM 降 batch）。

### 4.3 `16_submit_3_shipvit_infer.bat` — ship_reid 系成员推理

- 做什么：从 `repro_cfg.py` 导出的 `%TEMP%\shipreid_repro_env_vitsteps.txt` 逐行读取「成员名|config|ckpt|pred」，对每个成员用 `ship_reid_vit\venv` 运行 `inference.py --config <成员 config> --ckpt <成员 ckpt> --use_test_task --out_prediction <成员 pred> --topk %RRF_MEMBER_TOPK%`（纯特征推理）。
  - 当前预设（`VIT_N=7`）要推理 7 个成员：`ship_reid_vit_full`（`train_vit_full.yaml` + `outputs\full_vit\checkpoints\epoch_100.pth`）、`ship_reid_convnext_t_full`（`train_convnext_t_full.yaml` + `outputs\full_convnext_t\checkpoints\epoch_080.pth`）、`ship_reid_convnext_t_strong_cm_full`（`train_convnext_t_strong_cm_full.yaml` + `outputs\full_convnext_t_strong_cm\checkpoints\epoch_080.pth`）、`ship_reid_convnext_s_pure_full`（`train_convnext_s_pure_full.yaml` + `outputs\full_convnext_s_pure\checkpoints\epoch_040.pth`）、`ship_reid_swin_s_v2_full`（`train_swin_s_v2_full.yaml` + `outputs\full_swin_s_v2\checkpoints\epoch_060.pth`）、`ship_reid_dinov2_b_ep040` / `_ep060`（`train_dinov2_b.yaml` + `outputs\screen_dinov2_b\checkpoints\epoch_040.pth` / `epoch_060.pth`）。
  - `vit_config` 必须逐成员指定：`split_idx` 不在 checkpoint 的配置回填白名单内，缺了会按默认值推理。
  - **波次14 的 `ms` 变体（`ship_reid_vit_full_ms`）需额外 `--multiscale --ms_weights 0.7 0.2 0.1`，16 号脚本不会自动加**（该预设当前未启用）。
- **默认行为：逐成员复用——该成员的 `pred` 已存在则跳过、不重跑**（保证结果可比）。需要全部重跑时执行 `16_submit_3_shipvit_infer.bat force`。
- 输出：`sims\test_full\top100_*_full.json`（dino 为 `top100_dinov2_b_ep0XX_v2.json`）。
- 失败怎么办：报「venv not found」→ 跑 `ship_reid_vit\01_setup.bat`；报「checkpoint not found」→ 备齐该成员的 `vit_ckpt`。

### 4.4 `17_submit_4_rrf_fuse.bat` — RRF 融合

- 做什么：按 `reproduce.yaml` 当前预设读取各成员预测列表，`score(c)=Σ_m w_m/(k+rank_m(c))`（`k=10`），每成员暴露 top-100 排名，每 query 取 top-10；O2S 方向按 `per_direction` 覆盖 3 个成员权重。
- 输入：当前预设 11 个成员预测文件（8 个来自 `sims\test_full\`、`sims\test\top100_base75.json`、dino 两个 `_v2.json`）；query 集合一致性由脚本校验。
- 输出：工程根上级 `prediction.json`（`out_prediction: ../prediction.json`，即 RRF 基座，随后由 S5 + 闭配 α 覆盖）；工作副本 `submission_rrf.json`。
- 可覆盖：`17_submit_4_rrf_fuse.bat --preset <名>`，或 `--weights 1 1 1 1 0.5 1 0.75 1 1 1 1`（11 个值，个数须与当前预设成员数一致，按预设内顺序一一对应）。
- 失败怎么办：报成员预测缺失 → 回 15/16 号；报 query 集合不一致 → 某成员预测与 task.json 不匹配，重跑该成员推理。

### 4.5 `18_submit_5_verify.bat` — 提交格式终检

- 做什么：调用 `scripts\check_prediction.py` 逐条对照官方「直接判定无效」条件（文件名/编码、query 集合、每 query 恰好 10 个、候选去重且属 gallery、模态匹配、无额外字段、无 query 自身泄漏）。
- 输出：PASS/FAIL 报告，并回显 `reproduce.yaml` 的 `local_val` 基准（当前为 `N/A (blind)`）。
- 失败怎么办：**不要上传**；按 FAIL 明细回溯到 17 或成员推理。校验用显式报错而非 `assert`，`python -O` 下不会被剥离。

### 4.6 完整跑通顺序（端到端）

统一前提：所有脚本都在**工程根目录**执行（`ship_reid_vit\*` 除外，它们在自己目录内执行）；14~18 每一步都会重新解析 `configs\reproduce.yaml`，不依赖上一步残留的环境变量文件。

#### 4.6.1 从零首次复现（两个 venv 与所有产物都不存在）

| 序 | 脚本 | 运行环境 | 为什么必须在这个位置 |
|---|---|---|---|
| 1 | `ship_reid_vit\01_setup.bat` | 系统 Python（脚本自建 venv） | 建 `ship_reid_vit\venv`、下 `weights\vit_base_patch16_224.pth` |
| 2 | `ship_reid_vit\02_prepare_data.bat` | `ship_reid_vit\venv` | 产出 `question6-data\traindata\local_val_task.json` 与 `labels_train.csv`；**12 号第 6 步（HOSS 转换）硬依赖前者** |
| 3 | `12_sdfnet_setup.bat` | 系统 Python（脚本自建 venv） | 建 `.venv-sdfnet`、下载官方权重并派生 `SDF-Net_256.pth`、转 HOSS 数据 |
| 4 | `13_train_champion.bat` | 两个 venv（`ship_reid_vit\venv` + `.venv-sdfnet`）+ GPU | 重训冠军链的 8 个 full 成员（已有 ckpt 逐个跳过；首次完整训练约 6h）。**`sdfnet_base75` 与 dino 两成员不在本步骤内** |
| 5 | `14_submit_1_env_check.bat` | `.venv-sdfnet` | 自检，按 `[MISS]` 补齐后再继续 |
| 6 | `15_submit_2_sdfnet_infer.bat` | `.venv-sdfnet` | 3 个全量 SDF-Net 成员逐个推理 → `sims\test_full\top100_*.json` |
| 7 | `16_submit_3_shipvit_infer.bat` | `ship_reid_vit\venv` | 7 个 ship_reid 系成员 → `sims\test_full\top100_*.json` |
| 8 | `17_submit_4_rrf_fuse.bat` | `.venv-sdfnet` | RRF 融合 → 根目录 `prediction.json`（RRF 基座） |
| 9 | `18_submit_5_verify.bat` | `.venv-sdfnet` | 提交格式终检；PASS 才进入下一步 |
| 10 | `scripts\s5_rebuild.py`（见 §4.7） | `.venv-sdfnet` | S5 互反重排 → `prediction_s5.json` |
| 11 | `scripts\pair_match.py` + `pair_verify.py`（见 §4.7） | `.venv-sdfnet` | 闭配 α 簇修正 → 最终 `prediction.json`（Public 0.862） |

> 冠军链所需的训练权重（`outputs\full_*`、`outputs\screen_dinov2_b`、`logs\SDF-Net-full`、`logs\SDF-Net-mos-full`）由 **`13_train_champion.bat`** 生成（dino 两成员须另行由 `screen_dinov2_b` 训练流程产出，本仓库不再提供其训练脚本）。

`ship_reid_vit\04_local_val.bat` 与 `05_inference.bat` **不在提交链路内**：04 是 ship_reid_vit 侧的 legacy 本地验证自测（TTA + rerank + 三方向打分），05 是 legacy 旧推理入口；二者默认 ckpt `outputs\checkpoints\best.pth` **已在 2026-10-18 清理中删除**，需要时重跑 `ship_reid_vit\03_train.bat` 重建（**与冠军链无关**）。

#### 4.6.2 产物已就绪时的最短链路

若当前预设所需产物都已存在（两份官方权重、`SDF-Net\data\HOSS` 与 `data\full\HOSS`、`logs\SDF-Net-{full,mos-full}\transformer_*.pth`、5 个 ship_reid 全量权重 + 2 个 dino 权重、`sims\test_full\top100_*` 与 `sims\test\top100_base75.json`），则只剩两个虚拟环境需要重建：

```bat
ship_reid_vit\01_setup.bat       :: 建 ship_reid_vit\venv
12_sdfnet_setup.bat              :: 建 .venv-sdfnet；权重 / HOSS 数据检测到已存在会跳过
13_train_champion.bat            :: 8 个 full 成员冠军 ckpt 都已存在 -> 全部 skip
14_submit_1_env_check.bat        :: 预期全部 OK（0 MISS，退出码 0）
15_submit_2_sdfnet_infer.bat     :: 3 个全量成员重算
16_submit_3_shipvit_infer.bat    :: 逐成员检测到 pred 已存在 -> REUSED
17_submit_4_rrf_fuse.bat         :: -> prediction.json（RRF 基座）
18_submit_5_verify.bat
:: 再按 §4.7 叠 S5 与闭配 α 得到最终 0.862 提交
```

#### 4.6.3 必知约束

- **14 号要求两个 venv 与所有成员 ckpt 同时存在**：自检把「ship_reid venv python」与每个成员的 `vit_ckpt` 列为**必须项**（`scripts\repro_cfg.py::required_items`），缺一项即 MISS；成员**预测文件**不列为必须项，缺失只提示不阻断。`sdfnet_base75` 的 `ckpt` 已置空（外部权重在队友 D 盘），故当前预设**无 MISS**。
- **15/16/17/18 的第一件事都是拿 `.venv-sdfnet` 跑 `repro_cfg.py`**，所以 `12_sdfnet_setup.bat` 必须最先完成；即使 16 号最终逐成员走复用分支，也绕不开这一步。
- **冠军链的权重来源**：`ship_reid` 系成员的 `outputs\full_*\checkpoints\` 由 `13_train_champion.bat` 产出；dino 两成员由 `screen_dinov2_b` 流程产出（本仓库不再提供该训练脚本）。若任一权重缺失，当前预设会 MISS。

### 4.7 冠军链末段的 S5 与闭配 α（不在 14~18 号内）

```powershell
$R = "h:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main"
& "$R\.venv-sdfnet\Scripts\python.exe" "$R\scripts\s5_rebuild.py"                    # S5: RRF_base + 0.5·R2 互反重排
& "$R\.venv-sdfnet\Scripts\python.exe" "$R\scripts\pair_match.py"  --emit ...        # 闭配 α 簇修正
& "$R\.venv-sdfnet\Scripts\python.exe" "$R\scripts\pair_verify.py" ...
```

- `s5_rebuild.py --fidelity` 可重跑 267/267 逐位复现波次13 部署物（sha `237F1E77…`）。
- S5 用到的稠密 sim：`sims\test_rerank_full\sim_*.pt`（7 个 ship_reid 全量）+ `sims\test_rerank\sdfsim_{mos75,ep45,ep25}_full\sim.pt`（3 个 SDF-Net 全量）。`sdfnet_base75` 无稠密 sim（权重在队友 D 盘），仅经 RRF 基座贡献。
- 闭配 α 依赖 `scripts\pair_anchors.json`，诊断产物 `outputs\pair_diag.csv`。

---

## 5. 依赖关系图

```
                  12_sdfnet_setup.bat ─┐
        (建 .venv-sdfnet / 官方权重 / HOSS 数据)
                                       ▼
                    14_submit_1_env_check   [.venv-sdfnet]
                                       ▼
                    15_submit_2_sdfnet_infer [.venv-sdfnet] ──► sims\test_full\top100_{mos75,ep45,ep25}_full.json
                                       ▼
ship_reid_vit\01_setup.bat ──►  ship_reid_vit\venv
                                       ▼
                    16_submit_3_shipvit_infer [ship_reid_vit\venv] ──► sims\test_full\top100_shipvit_full.json
                                                                       sims\test_full\top100_convnext_t_ep080_full.json
                                                                       sims\test_full\top100_convnext_t_strong_cm_full.json
                                                                       sims\test_full\top100_convnext_s_pure_ep040_full.json
                                                                       sims\test_full\top100_swin_s_v2_ep060_full.json
                                                                       sims\test_full\top100_dinov2_b_ep0{40,60}_v2.json
                                       ▼
                    17_submit_4_rrf_fuse     [.venv-sdfnet] ──► prediction.json（RRF 基座）
                                       ▼
                    18_submit_5_verify       [.venv-sdfnet]
                                       ▼
        s5_rebuild.py（S5 互反重排） ──► prediction_s5.json
                                       ▼
        pair_match.py + pair_verify.py（闭配 α） ──► prediction.json（Public 0.862）
```

- 14 / 15 / 17 / 18 号**必须用 `.venv-sdfnet`**。
- **16 号是唯一需要 `ship_reid_vit\venv` 的步骤**（若该预设所有 ship_reid 系成员的 `pred` 都已存在，则逐一跳过，不需要该环境）。

---

## 6. 常见问题 / 注意事项

| 事项 | 说明 |
|---|---|
| 16 号默认复用 | 逐个 ship_reid 系成员：其 `pred` 已存在时不重跑；要全部重跑加 `force` |
| 15 号缺 ckpt 跳过 | 某 SDF-Net 成员 ckpt 不在本机但 pred 已存在 -> 打印 skip 并复用。当前预设 `sdfnet_base75` 的 `ckpt` 已置空（不进推理步骤），不阻断本机链路 |
| Git | 工程根目录含 `.git`（历史提交只读）；无指令不做新提交，删除/改动前先备份 |
| 路径统一 | 所有相对路径一律相对工程根目录解析（`scripts\repro_cfg.py::resolve`）；bat 里不写死路径 |
| 唯一事实来源 | 改成员/权重/k/方向覆盖只改 `configs\reproduce.yaml` 的 `ensemble.members` / `ensemble.presets` / `ensemble.active` / `k` / `topk` / `member_topk` |
| 成员定义 | `{name, ckpt, pred}`；`ckpt` 相对 `sdf_checkpoint_dir`（留空 = 不做 SDF-Net 本地推理）；`pred` 相对工程根目录。ship_reid 系成员另加 `vit_config` + `vit_ckpt`（均相对 `shipvit_dir`，由 16 号自动推理；`vit_config` 必须逐成员写，因 `split_idx` 不在 ckpt 配置回填白名单内） |
| 方向覆盖 | 预设内可选键 `per_direction: {O2S\|S2O\|O2O: {成员: 权重}}`；仅覆盖列出的成员，0 = 该方向剔除该成员。当前 active 仅覆盖 O2S |
| 提交文件 | 最终 = 工程根上级 `prediction.json`（**Public 0.862**，闭配 α 版）；`submission_rrf.json` 是 RRF 基座工作副本 |
| 冠军 vs 回滚 | 冠军 = active 预设（11 成员）+ S5 + 闭配 α；`prediction_s5.json` 为 S5 基座回滚点，`prediction_ag_bak.json` 为 α+γ（0.856）备份 |
| 选模 | 用 `scripts\sdfnet_ckpt_sweep.py` 按 `Final` 排序；**严禁**按 SDF-Net 内置 val mAP / overall mAP 选模 |
| 保留门槛 | 每新增模块若 `Final` 提升 < 0.003 则删除；多成员选择必须做 fold 交叉验证（两折同时 ≥0.003 才保留） |
| 全量成员的盲提交约束 | 全量（3534 身份）重训成员已见全部身份 -> 其上任何新成员/新权重都**无无泄漏验证**，本地三重门槛只对「同基座、同泄漏状态」成立（exp_008 / 波次12 双重证据：盲链上叠盲成员 = 官方判负） |
| 训练超参 | AdamW、`BASE_LR 1e-4`、`IMS_PER_BATCH 32`、`NUM_INSTANCE 2`、`MAX_EPOCHS 80`、每 5 epoch 存 ckpt；调度为 CosineLRScheduler（`t_initial=SOLVER.MAX_EPOCHS`，`SOLVER.STEPS` 实际未使用，延长训练只能改 `MAX_EPOCHS`） |
| 官方权重两份 | `SDF-Net.pth` 是官方 256×128 原始权重；`SDF-Net_256.pth` 是它的 256×256 衍生版（pos_embed 插值）。链路只用后者，两份均由 `12_sdfnet_setup.bat` 自动准备 |
| 历史预设不可重跑 | 清理后 `final` / `final_full` / `final_full_b75` / `..._ms` / `..._dino_full_...` 等预设引用的 2828 时代 ckpt 已删除；`reproduce.yaml` 保留它们仅为历史记录 |
| 官方模板 | `README.official.md` / `技术方案.official.md` / `路线方案.official.md` 为赛题官方起始模板（历史存档，已过期，仅参照） |

---

## 7. 参考文档

| 文档 | 内容 |
|---|---|
| `技术方案.md` | 任务与数据、评测协议与铁律、骨干/异构分支/RRF 设计、量化结果、CMAL 损失说明 |
| `路线方案.md` | 约束锚点、已完成阶段结论、负结果登记表、剩余风险、P0 优化项与下一步方向 |
| `RRF集成方案.md` | RRF 的推导与可靠性说明（历史数值口径以 `reproduce.yaml` 为准） |
| `优化路线1.md` | 早期优化路线构想的存档（历史，引用已删除的 `ship_reid_system\`） |
| `configs\reproduce.yaml` | 路径与集成配方的唯一事实来源 |
| `experiments\` | exp_001~007 / exp_010 实验登记（含已否决探针及其清理记录） |
| `README.official.md` / `技术方案.official.md` / `路线方案.official.md` | 赛题官方起始模板（历史存档，已过期） |
*（内容由AI生成，仅供参考）*
