# 赛题06 多源卫星跨模态舰船重识别 · 复现指南

> **一句话**：光学（Optical）与 SAR 两种模态下同一艘舰船的跨模态身份匹配。最终方案 = **RRF 倒数排名融合**（3 个 SDF-Net checkpoint + 5 个异构分支：ViT / ConvNeXt-Tiny / ConvNeXt-Small×2 / Swin-Small），成员排名深度 top-100、`k=3`。**当前冠军 = 波次5 全量重训版（`ensemble.active: final_full`，盲提交）**：8 个成员改用官方全量 `labels.csv`（3534 身份/7422 图）重训，因成员已见全部身份而**无无泄漏本地验证**，增益靠 exp_005 受控嵌套实验外推 +0.02~+0.04（相对回滚基线 0.6905）。**回滚基线 = `ensemble.active: final`**（2828 身份缩水数据版，本地验证 Final **0.6905**：fold0 0.7046 / fold1 0.6764；单模型冠军基线 E0 = 0.5888）。

本文档面向「能在本机把最终提交复现出来」的人。技术选型理由见 `技术方案.md`，历史取舍与后续方向见 `路线方案.md`。

---

## 1. 当前状态

| 项 | 值 |
|---|---|
| 赛题 | 2026 全国大数据与计算智能挑战赛 · 赛题06「多源卫星跨模态舰船重识别」 |
| 测试集 | 267 query / 1800 gallery（O2S 120 / S2O 120 / O2O 27） |
| 选模判据 | `Final = 0.45×O2S + 0.45×S2O + 0.10×O2O`（方向分 = (R@1 + mAP@10)/2） |
| 冠军方案 | RRF 融合，`configs/reproduce.yaml` 的 `ensemble.active: final_full`（波次5 全量重训版，盲提交） |
| 回滚基线 | `ensemble.active: final`（缩水数据版，本地验证 Final 0.6905 / fold0 0.7046 / fold1 0.6764） |
| 本地验证 Final | 冠军 `final_full` **N/A（盲提交：成员训练已见全部 3534 身份，不存在无泄漏验证）**；回滚基线 0.6905 |
| 单模型冠军基线 | E0 = SDF-Net ep80：0.5888 |
| 最终提交文件 | `h:\Ship-Re-Identification\prediction.json` |
| 项目 Git | 工程根目录含 `.git`（历史提交只读，无指令不做新提交）；任何改动/删除前先备份 |

---

## 2. 目录结构（与磁盘一致）

工程根目录 = `h:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\`（已扁平化，**没有** `ship_reid_system/` 这一层）。

```
h:\Ship-Re-Identification\
├── prediction.json                     ★ 最终提交文件（267 query, top-10）
├── question6-data\                     赛题数据（只读）
│   ├── traindata\                      labels.csv / labels_train.csv / local_val_task.json / local_val_gt.json
│   └── preliminary-round-test-data\    task.json（初赛测试集）
└── Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\   ← 工程根目录
    ├── 12_sdfnet_setup.bat             环境准备（venv / 依赖 / 官方权重 / HOSS 数据）
    ├── 13_train_champion.bat           重训冠军 final_full 的 8 个成员（逐成员断点跳过）
    ├── 14_submit_1_env_check.bat       ① 环境自检
    ├── 15_submit_2_sdfnet_infer.bat    ② SDF-Net 成员推理
    ├── 16_submit_3_shipvit_infer.bat   ③ ship_reid 系成员推理（ViT / ConvNeXt-T/S / Swin-S，共 5 个）
    ├── 17_submit_4_rrf_fuse.bat        ④ RRF 融合
    ├── 18_submit_5_verify.bat          ⑤ 提交格式终检
    ├── configs\
    │   └── reproduce.yaml              ★ 唯一事实来源（路径 / 成员池 / 权重预设 / k）
    ├── scripts\
    │   ├── repro_cfg.py                解析 reproduce.yaml（相对路径均相对工程根目录）
    │   ├── rrf_fuse.py                 RRF 融合 + 平台级格式校验
    │   ├── sdfnet_inference.py         单模型推理 + 按 query_type 分组后处理
    │   ├── sdfnet_ckpt_sweep.py        checkpoint 选优（按 Final 排序）
    │   ├── sdfnet_prepare_data.py      赛题数据 → HOSS 布局
    │   ├── sdfnet_prepare_full.py      全量数据 → HOSS 布局（波次5，data/full）
    │   ├── sdfnet_download_weights.py  官方权重下载
    │   ├── check_prediction.py         提交格式独立校验器
    │   └── build_local_val.py / compute_stats.py
    ├── config\  data\  models\  inference\  utils\   自研模块（保留）
    ├── evaluate.py                     赛题提交评测入口（--submission）
    ├── SDF-Net\                        主骨干（上游 cfrfree/SDF-Net + 本项目改动）
    │   ├── configs\SDF-Net.yml         E0 微调配置
    │   ├── configs\SDF-Net-full.yml    波次5 全量重训配置（唯一差异：DATASETS.ROOT_DIR=data/full）
    │   ├── configs\SDF-Net-mos.yml     MOS 重训配置（唯一差异：CMAL_LOSS_WEIGHT=1.0）
    │   ├── configs\SDF-Net-mos-full.yml
    │   ├── loss\cmal_loss.py           CMAL 对齐损失
    │   └── ...
    ├── ship_reid_vit\                  异构分支（ViT / ConvNeXt-T/S / Swin-S 双分支，与 SDF-Net 并列）
    │   ├── 01_setup.bat … 05_inference.bat
    │   ├── config\train_vit.yaml                 成员 ship_reid_vit 的推理配置
    │   ├── config\train_convnext_t.yaml          成员 ship_reid_convnext_t 的推理配置
    │   ├── config\train_convnext_s.yaml          成员 ship_reid_convnext_s 的推理配置
    │   ├── config\train_convnext_s_pure.yaml     成员 ship_reid_convnext_s_pure 的推理配置（波次4 强配方）
    │   ├── config\train_swin_s_v2.yaml           成员 ship_reid_swin_s_v2 的推理配置（波次4 强配方）
    │   ├── config\train_*_full.yaml              上述 5 个成员的波次5 全量重训配置（train_labels_csv=labels.csv）
    │   ├── outputs\checkpoints\best.pth                              ★ final 预设 ship_reid_vit 权重
    │   ├── outputs\screen_convnext_t\checkpoints\epoch_080.pth       ★ final 预设 convnext_t 权重
    │   ├── outputs\screen_convnext_s\checkpoints\epoch_060.pth       ★ final 预设 convnext_s 权重
    │   ├── outputs\screen_convnext_s_pure\checkpoints\epoch_040.pth  ★ final 预设 convnext_s_pure 权重
    │   ├── outputs\screen_swin_s_v2\checkpoints\epoch_060.pth        ★ final 预设 swin_s_v2 权重
    │   ├── outputs\full_vit\checkpoints\epoch_100.pth                ★ final_full 预设 ship_reid_vit_full 权重
    │   ├── outputs\full_convnext_t\checkpoints\epoch_080.pth         ★ final_full 预设 convnext_t_full 权重
    │   ├── outputs\full_convnext_s\checkpoints\epoch_060.pth         ★ final_full 预设 convnext_s_full 权重
    │   ├── outputs\full_convnext_s_pure\checkpoints\epoch_040.pth    ★ final_full 预设 convnext_s_pure_full 权重
    │   ├── outputs\full_swin_s_v2\checkpoints\epoch_060.pth          ★ final_full 预设 swin_s_v2_full 权重
    │   └── weights\vit_base_patch16_224.pth
    ├── logs\
    │   ├── SDF-Net-finetune\           E0 微调（transformer_5…80.pth） ★ final 回滚预设 ep45/ep25 来源
    │   ├── SDF-Net-mos\                MOS 重训（transformer_75.pth） ★ final 回滚预设锚点来源
    │   ├── SDF-Net-full\               波次5 全量（transformer_25/45.pth） ★ 成员 sdfnet_ep25/45_full 来源
    │   └── SDF-Net-mos-full\           波次5 全量 MOS（transformer_75.pth） ★ 成员 sdfnet_mos75_full 来源
    ├── sims\
    │   ├── test_full\top100_*_full.json   波次5 全量成员测试集预测（冠军 final_full 用，15/16 号产出）
    │   └── test\top100_*.json             缩水数据成员测试集预测（回滚预设 final 用，15/16 号产出）
    ├── experiments\                    实验记录（exp_001~007，含已否决探针）
    └── *.official.md                   赛题官方起始模板（只读，已过期）
```

---

## 3. 环境准备

需要**两个**独立虚拟环境，当前两者都已被删除，端到端复现前必须重建。

### 3.1 `.venv-sdfnet`（14~18 号脚本使用）

```bat
:: 在工程根目录双击或执行：
12_sdfnet_setup.bat
```

该脚本幂等地完成：
1. 创建 `.venv-sdfnet`（Python 3.9+）
2. 安装 `torch==2.2.2+cu118` / `torchvision==0.17.2+cu118` + `numpy/opencv-python/Pillow/thop/timm/yacs`（无 NVIDIA GPU 时脚本会提示改用 CPU 版）
3. 克隆上游 `SDF-Net`（已存在且含 `train.py` 则跳过）
4. 从 HuggingFace `Chenfree233/SDF-Net` 下载官方权重 → `SDF-Net\logs\SDF-Net\SDF-Net.pth`（约 333MB）
5. 派生 256×256 版本 → `SDF-Net\logs\SDF-Net\SDF-Net_256.pth`：官方权重是在 256×128 下训练的（`pos_embed` 为 `[1,130,768]`），直接配 256×256 输入会形状不匹配，故调 `scripts\sdfnet_resize_posembed.py` 把 pos_embed 双线性插值到 `[1,258,768]`（插值期间 cls / spe 两个前缀 token 原样保留）
6. 把赛题数据转成 HOSS 布局 `SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}`，**复用现有 `local_val_task.json`，不重新划分**

可用 `SDF_DATA_DIR` 覆盖数据目录。12 号结束后直接进入 §4（从 14 号开始）。

> **两个权重文件的分工**：`SDF-Net.pth` 是官方原始 256×128 权重（下载产物）；**`SDF-Net_256.pth` 才是本项目 256×256 链路实际使用的那份**——`reproduce.yaml` 的 `sdfnet_official` 与 `SDF-Net.yml` / `SDF-Net-mos.yml` 的 `PRETRAIN_PATH`、`WEIGHT` 均指向它，14 号自检检查的也是它。两者都由 12 号自动准备完成，无需手工干预。

> **波次5 全量数据目录**：冠军 `final_full` 的 3 个 SDF-Net 成员用 `SDF-Net\data\full\HOSS\`（由 `scripts\sdfnet_prepare_full.py` 生成，训练集为全量 `labels.csv` 的 7422 图；query/gallery 以 junction 复用现役目录）。该目录与 `data\HOSS` 平级，不在 12 号默认流程内。

### 3.2 `ship_reid_vit\venv`（仅 16 号 ship_reid 系成员推理使用）

```bat
:: 进入 ship_reid_vit 目录后执行：
01_setup.bat
```

创建 `ship_reid_vit\venv`、安装 torch/torchvision（CUDA 12.4，失败回落 CPU）+ `requirements.txt`，并下载 `weights\vit_base_patch16_224.pth`。

> 注意：`ship_reid_vit` 的训练（`03_train.bat`）会重新生成 `outputs\checkpoints\best.pth`；若该权重已存在且无需重训，可跳过训练直接进入 §4。

---

## 4. bat 脚本（12、13、14~18）与提交流程

### 4.0 全部 bat 一览（12、13、14~18）

| # | 脚本 | 定位 | 做什么 | 运行环境 | 主要输入 → 输出 |
|---|---|---|---|---|---|
| 12 | `12_sdfnet_setup.bat` | 环境准备（一次性） | 建 `.venv-sdfnet`；装 `torch 2.2.2+cu118` / `torchvision 0.17.2+cu118` + `numpy/opencv-python/Pillow/thop/timm/yacs`；克隆上游 SDF-Net（已存在且含 `train.py` 则跳过）；从 HuggingFace `Chenfree233/SDF-Net` 下载官方权重 `SDF-Net.pth`（约 333MB）；**派生 256×256 版本 `SDF-Net_256.pth`**（pos_embed 插值，链路真正使用的那份）；把赛题数据转成 HOSS 布局 | 系统 Python（脚本自建 venv） | `question6-data\traindata\` → `.venv-sdfnet\`、`SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}`、`SDF-Net\logs\SDF-Net\SDF-Net.pth` + `SDF-Net_256.pth` |
| 13 | `13_train_champion.bat` | 训练（一次性，可断点续跑） | 重训冠军 `final_full` 的 8 个成员：5 个 ship_reid 全量配置（`train_{vit,convnext_t,convnext_s,convnext_s_pure,swin_s_v2}_full.yaml`）+ 2 个 SDF-Net 全量配置（`SDF-Net-full.yml` / `SDF-Net-mos-full.yml`）。逐成员按「冠军 checkpoint 是否已存在」跳过；ship_reid 成员若 `outputs\<dir>\last.pth` 存在则自动加 `--resume auto`；SDF-Net 全量训练数据 `data/full/HOSS` 缺失时自动跑 `sdfnet_prepare_full.py` 生成 | 两个 venv：`ship_reid_vit\venv`（5 个 ship_reid）+ `.venv-sdfnet`（2 个 SDF-Net） | `labels.csv` + 5 个 `train_*_full.yaml` + `SDF-Net-{full,mos-full}.yml` → `ship_reid_vit\outputs\full_*\checkpoints\epoch_*.pth`、`logs\SDF-Net-{full,mos-full}\transformer_*.pth` |
| 14 | `14_submit_1_env_check.bat` | **提交链 ①** | 解析 `reproduce.yaml` 把路径导出为环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项自检当前预设所需输入是否齐备 | `.venv-sdfnet` | `configs\reproduce.yaml` → 控制台 OK/MISS 清单；齐备退出码 0 |
| 15 | `15_submit_2_sdfnet_infer.bat` | **提交链 ②** | 对当前预设中**带 `ckpt` 的 SDF-Net 成员**逐个在测试集推理（cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）；每成员输出 `--topk %RRF_MEMBER_TOPK%` 个候选（当前 100）；`SDF_N=0` 时直接跳过 | `.venv-sdfnet` | `final_full`：`logs\SDF-Net-mos-full\transformer_75.pth`、`logs\SDF-Net-full\transformer_45/25.pth` → `sims\test_full\top100_{mos75,ep45,ep25}_full.json`；`final`：`logs\SDF-Net-mos\` + `logs\SDF-Net-finetune\` → `sims\test\top100_{mos75,ep45,ep25}.json` |
| 16 | `16_submit_3_shipvit_infer.bat` | **提交链 ③** | 遍历当前预设中**声明了 `vit_ckpt` 的 ship_reid 系成员**（ViT / ConvNeXt-Tiny / ConvNeXt-Small / ConvNeXt-Small 强配方 / Swin-Small 强配方，共 5 个），逐个用 `inference.py --tta --rerank` 推理。**逐成员复用：该成员的 `pred` 已存在则跳过、不重跑**（保证结果可比）；要重跑加参数 `force`。配置与权重来自成员自身的 `vit_config` / `vit_ckpt`（相对 `ship_reid_vit`），不是硬编码 | `ship_reid_vit\venv` | 5 组 config + ckpt（`final_full` 用 `train_*_full.yaml` + `outputs\full_*\checkpoints\epoch_*.pth`；`final` 用 `train_*.yaml` + `outputs\screen_*\checkpoints\epoch_*.pth`）→ `sims\test_full\top100_*_full.json` 或 `sims\test\top100_*.json` |
| 17 | `17_submit_4_rrf_fuse.bat` | **提交链 ④** | RRF 倒数排名融合：`score(c) = Σ_m w_m / (k + rank_m(c))`，`k=3`、每成员暴露 top-100 排名、每 query 取 top-10；成员与权重在 Python 侧按序绑定（不经过 cmd） | `.venv-sdfnet` | 全部成员预测（`final_full` / `final` 均 8 个文件） → `h:\Ship-Re-Identification\prediction.json`（工作副本 `submission_rrf.json`） |
| 18 | `18_submit_5_verify.bat` | **提交链 ⑤** | 调 `scripts\check_prediction.py` 逐条对照平台「直接判定无效」条件核验提交文件，并回显本地验证基准 | `.venv-sdfnet` | `prediction.json` + `task.json` → PASS/FAIL 报告（`final_full` 显示 N/A（盲提交）；`final` 显示 0.6905 / 0.7046 / 0.6764） |

> 分工：**14→15→16→17→18** 是唯一的提交链路（下文 §4.1~§4.5 逐步展开）；**12** 是一次性环境准备；**13** 是一次性权重重训（仅在需要重建冠军成员权重时跑，已有 ckpt 会逐个跳过）。**端到端从零到提交的完整顺序见 §4.6。**

统一约定：所有步骤在**工程根目录**执行；每一步都会重新解析 `configs/reproduce.yaml`（不读取陈旧的环境变量文件）。产物链路（以当前冠军 `final_full` 为例）：

```
成员预测（SDF-Net 侧 sims/test_full/top100_mos75_full.json / top100_ep45_full.json / top100_ep25_full.json
          + ship_reid 侧 top100_shipvit_full.json / top100_convnext_t_ep080_full.json / top100_convnext_s_ep060_full.json
                       / top100_convnext_s_pure_ep040_full.json / top100_swin_s_v2_ep060_full.json）
        └──────────────► 17 号 RRF 融合 ──► ../prediction.json（最终提交）
```

### 4.1 `14_submit_1_env_check.bat` — 环境自检

- 做什么：解析 `reproduce.yaml` → 导出环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项检查当前预设（`final_full`）所需输入是否齐备。
- 输入：`.venv-sdfnet`、`SDF-Net.yml`、官方权重、赛题 `task.json`、各 SDF-Net 成员 ckpt（有 `ckpt` 字段的成员）、ship_reid 系成员的 venv 与各自 `vit_ckpt`（共 5 个：`final_full` 为 `outputs\full_*\checkpoints\epoch_*.pth`；`final` 为 `best.pth`、`screen_convnext_t\…epoch_080.pth`、`screen_convnext_s\…epoch_060.pth`、`screen_convnext_s_pure\…epoch_040.pth`、`screen_swin_s_v2\…epoch_060.pth`）。
- 输出：控制台 OK/MISS 清单；全部必需项齐备时退出码 0。**成员预测文件不算必需项**（由 15/16 号本地推理产出），缺失只显示为警告，不阻断。
- 失败怎么办：按 `[MISS]` 行补齐——venv 缺失回 §3；ckpt/official 缺失跑 `12_sdfnet_setup.bat`；ship_reid 侧缺失跑 `ship_reid_vit\01_setup.bat` 或备齐对应 `vit_ckpt`。
- 可用 `--preset <名>` 临时切换预设做自检（如 `--preset final` 切回滚基线）。

### 4.2 `15_submit_2_sdfnet_infer.bat` — SDF-Net 成员推理

- 做什么：对当前预设中**带 `ckpt` 的成员**逐个在测试集推理（`scripts\sdfnet_inference.py`，`--batch_size 16`，`--topk %RRF_MEMBER_TOPK%`，cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）。
- 输入（`final_full`）：`logs\SDF-Net-mos-full\transformer_75.pth`（锚点）、`logs\SDF-Net-full\transformer_45.pth`、`transformer_25.pth`；`question6-data\...\task.json`。
- 输出（`final_full`）：`sims\test_full\top100_mos75_full.json` / `top100_ep45_full.json` / `top100_ep25_full.json`（共 3 个成员需本地推理；`SDF_N=0` 时直接跳过）。
- 失败怎么办：检查 ckpt 路径（`ckpt` 相对 `sdf_checkpoint_dir`，`*_full` 成员在 `reproduce.yaml` 里写绝对路径）、显存（OOM 降 batch）。

### 4.3 `16_submit_3_shipvit_infer.bat` — ship_reid 系成员推理

- 做什么：`final_full` / `final` / `anchor_plus_sv` 预设都含 ship_reid 系成员。脚本从 `repro_cfg.py` 导出的 `%TEMP%\shipreid_repro_env_vitsteps.txt` 逐行读取「成员名|config|ckpt|pred」，对每个成员用 `ship_reid_vit\venv` 运行 `inference.py --config <成员 config> --ckpt <成员 ckpt> --use_test_task --out_prediction <成员 pred> --topk %RRF_MEMBER_TOPK% --tta --rerank`。**配置与权重来自成员自身的 `vit_config` / `vit_ckpt`**（相对 `ship_reid_vit`），因此新增 ship_reid 系成员只需在 `reproduce.yaml` 里声明，无需改本脚本。
  - `final_full` 预设下要推理 5 个成员：`ship_reid_vit_full`（`train_vit_full.yaml` + `outputs\full_vit\checkpoints\epoch_100.pth`）、`ship_reid_convnext_t_full`（`train_convnext_t_full.yaml` + `outputs\full_convnext_t\checkpoints\epoch_080.pth`）、`ship_reid_convnext_s_full`（`train_convnext_s_full.yaml` + `outputs\full_convnext_s\checkpoints\epoch_060.pth`）、`ship_reid_convnext_s_pure_full`（`train_convnext_s_pure_full.yaml` + `outputs\full_convnext_s_pure\checkpoints\epoch_040.pth`）、`ship_reid_swin_s_v2_full`（`train_swin_s_v2_full.yaml` + `outputs\full_swin_s_v2\checkpoints\epoch_060.pth`）。
  - `final` 预设下同理 5 个成员，配置为 `train_{vit,convnext_t,convnext_s,convnext_s_pure,swin_s_v2}.yaml`，权重为 `outputs\checkpoints\best.pth` 与 `outputs\screen_*\checkpoints\epoch_*.pth`。
  - `vit_config` 必须逐成员指定：`split_idx` 不在 checkpoint 的配置回填白名单内，缺了会按默认值推理。
- **默认行为：逐成员复用——该成员的 `pred` 已存在则跳过、不重跑**（保证结果可比）。需要全部重跑时执行 `16_submit_3_shipvit_infer.bat force`。
- 输出：`sims\test_full\top100_*_full.json`（`final_full`）或 `sims\test\top100_*.json`（`final`）（`VIT_N=0` 时直接跳过）。
- 失败怎么办：报「venv not found」→ 跑 `ship_reid_vit\01_setup.bat`；报「checkpoint not found」→ 备齐该成员的 `vit_ckpt`。

### 4.4 `17_submit_4_rrf_fuse.bat` — RRF 融合

- 做什么：按 `reproduce.yaml` 当前预设读取各成员预测列表，`score(c)=Σ_m w_m/(k+rank_m(c))`（`k=3`），每成员暴露 top-100 排名，每 query 取 top-10。
- 输入：全部成员预测（`final_full` 预设 8 个文件，全部来自 `sims\test_full\`）；query 集合一致性由脚本校验。
- 输出：`h:\Ship-Re-Identification\prediction.json`（`out_prediction: ../prediction.json`）；工作副本 `submission_rrf.json`。
- 可覆盖：`17_submit_4_rrf_fuse.bat --preset final`，或 `--weights 1 1 1 1 1 0.5 0.75 1`（个数须与当前预设成员数一致，按预设内顺序一一对应）。
- 失败怎么办：报成员预测缺失 → 回 15/16 号；报 query 集合不一致 → 某成员预测与 task.json 不匹配，重跑该成员推理。

### 4.5 `18_submit_5_verify.bat` — 提交格式终检

- 做什么：调用 `scripts\check_prediction.py` 逐条对照官方「直接判定无效」条件（文件名/编码、query 集合、每 query 恰好 10 个、候选去重且属 gallery、模态匹配、无额外字段、无 query 自身泄漏）。
- 输出：PASS/FAIL 报告，并回显本地验证基准（`final_full` = N/A（盲提交）；`final` = Final 0.6905 / fold0 0.7046 / fold1 0.6764）。
- 失败怎么办：**不要上传**；按 FAIL 明细回溯到 17 或成员推理。校验用显式报错而非 `assert`，`python -O` 下不会被剥离。

### 4.6 完整跑通顺序（端到端）

统一前提：所有脚本都在**工程根目录**执行（`ship_reid_vit\*` 除外，它们在自己目录内执行）；14~18 每一步都会重新解析 `configs\reproduce.yaml`，不依赖上一步残留的环境变量文件。

#### 4.6.1 从零首次复现（两个 venv 与所有产物都不存在）

| 序 | 脚本 | 运行环境 | 为什么必须在这个位置 |
|---|---|---|---|
| 1 | `ship_reid_vit\01_setup.bat` | 系统 Python（脚本自建 venv） | 建 `ship_reid_vit\venv`、下 `weights\vit_base_patch16_224.pth` |
| 2 | `ship_reid_vit\02_prepare_data.bat` | `ship_reid_vit\venv` | 产出 `question6-data\traindata\local_val_task.json` 与 `labels_train.csv`；**12 号第 6 步（HOSS 转换）硬依赖前者** |
| 3 | `ship_reid_vit\03_train.bat` | `ship_reid_vit\venv` + GPU | 产出 `outputs\checkpoints\best.pth`（16 号推理 `ship_reid_vit` 的输入） |
| 4 | `12_sdfnet_setup.bat` | 系统 Python（脚本自建 venv） | 建 `.venv-sdfnet`、下载官方权重并派生 `SDF-Net_256.pth`、转 HOSS 数据 |
| 5 | `13_train_champion.bat` | 两个 venv（`ship_reid_vit\venv` + `.venv-sdfnet`）+ GPU | 重训冠军 `final_full` 的 8 个成员（已有 ckpt 逐个跳过；首次完整训练约 6h）。`final` 回滚预设的缩水数据权重不在本步骤内 |
| 6 | `14_submit_1_env_check.bat` | `.venv-sdfnet` | 自检，按 `[MISS]` 补齐后再继续 |
| 7 | `15_submit_2_sdfnet_infer.bat` | `.venv-sdfnet` | 3 个 SDF-Net 成员逐个推理 → `sims\<SDF_SIM_DIR>\top100_*.json` |
| 8 | `16_submit_3_shipvit_infer.bat` | `.venv-sdfnet`（`ship_reid_vit\venv` 仅当需要重跑推理） | 5 个 ship_reid 系成员 → `sims\<VIT_PRED_DIR>\top100_*.json` |
| 9 | `17_submit_4_rrf_fuse.bat` | `.venv-sdfnet` | RRF 融合 → `../prediction.json` |
| 10 | `18_submit_5_verify.bat` | `.venv-sdfnet` | 提交格式终检；PASS 才上传 |

> `final_full` 所需的全部训练权重（`outputs\full_*`、`logs\SDF-Net-full`、`logs\SDF-Net-mos-full`）由 **`13_train_champion.bat`** 生成（或须预先存在）；它们不在 01~12 链路内。`final` 回滚预设的缩水数据权重（`outputs\screen_*`、`outputs\checkpoints`、`logs\SDF-Net-finetune`、`logs\SDF-Net-mos`）同理。

`04_local_val.bat` 与 `05_inference.bat` **不在提交链路内**：04 是 ship_reid_vit 侧的本地验证自测（TTA + rerank + 三方向打分）；**05 与 16 号对 `ship_reid_vit` 成员是同一条命令**（`inference.py --config config\train_vit.yaml --use_test_task --tta --rerank`），只是 16 号已泛化为从 `reproduce.yaml` 遍历所有 ship_reid 系成员、并按 `pred` 路径落盘。

#### 4.6.2 产物已就绪时的最短链路

若当前预设（`final_full`）所需产物都已存在：两份官方权重、`SDF-Net\data\HOSS` 与 `data\full\HOSS`、`logs\SDF-Net-{full,mos-full}\transformer_*.pth`、5 个 ship_reid 系全量权重（`outputs\full_*\checkpoints\epoch_*.pth`）、`sims\test_full\top100_*_full.json`，则只剩两个虚拟环境需要重建：

```bat
ship_reid_vit\01_setup.bat       :: 建 ship_reid_vit\venv
12_sdfnet_setup.bat              :: 建 .venv-sdfnet；权重 / HOSS 数据检测到已存在会跳过
13_train_champion.bat            :: 8 个成员冠军 ckpt 都已存在 -> 全部 skip（可当产物完整性自检）
14_submit_1_env_check.bat
15_submit_2_sdfnet_infer.bat     :: 无复用逻辑，3 个成员会全部重算
16_submit_3_shipvit_infer.bat    :: 逐个成员检测到 top100_*_full.json 已存在 -> REUSED，不重跑
17_submit_4_rrf_fuse.bat
18_submit_5_verify.bat
```

#### 4.6.3 必知约束

- **14 号要求两个 venv 与所有成员 ckpt 同时存在**：当前预设含 ship_reid 系成员，自检会把「ship_reid venv python」与每个成员的 `vit_ckpt` 列为**必须项**（`scripts\repro_cfg.py::required_items`），缺一项即 MISS；成员**预测文件**不列为必须项，缺失只提示不阻断。
- **15/16/17/18 的第一件事都是拿 `.venv-sdfnet` 跑 `repro_cfg.py`**，所以 `12_sdfnet_setup.bat` 必须最先完成；即使 16 号最终逐成员走复用分支，也绕不开这一步。
- **骨干筛选权重不在 01~05 链路内**：`final` 回滚预设的 4 个骨干成员（`ship_reid_convnext_t` / `convnext_s` / `convnext_s_pure` / `swin_s_v2`）由骨干筛选专用配置训练，落在 `ship_reid_vit\outputs\screen_*\checkpoints\`；`final_full` 的对应成员落在 `outputs\full_*\checkpoints\`。若任一权重缺失，当前预设会 MISS，要么补齐该权重，要么把 `ensemble.active` 切到不含它的预设。

---

## 5. 依赖关系图

```
                  12_sdfnet_setup.bat ─┐
        (建 .venv-sdfnet / 官方权重 / HOSS 数据)
                                       ▼
                    14_submit_1_env_check   [.venv-sdfnet]
                                       ▼
                    15_submit_2_sdfnet_infer [.venv-sdfnet] ──► sims\<SDF_SIM_DIR>\top100_{mos75,ep45,ep25}[_full].json
                                       ▼
ship_reid_vit\01_setup.bat ──►  ship_reid_vit\venv
                                       ▼
                    16_submit_3_shipvit_infer [ship_reid_vit\venv] ──► sims\<..>\top100_shipvit[_full].json
                                                                        sims\<..>\top100_convnext_t_ep080[_full].json
                                                                        sims\<..>\top100_convnext_s_ep060[_full].json
                                                                        sims\<..>\top100_convnext_s_pure_ep040[_full].json
                                                                        sims\<..>\top100_swin_s_v2_ep060[_full].json
                                       ▼
                    17_submit_4_rrf_fuse     [.venv-sdfnet] ──► prediction.json
                                       ▼
                    18_submit_5_verify       [.venv-sdfnet]
```

- 14 / 15 / 17 / 18 号**必须用 `.venv-sdfnet`**。
- **16 号是唯一需要 `ship_reid_vit\venv` 的步骤**（若该预设所有 ship_reid 系成员的 `pred` 都已存在，则逐一跳过，不需要该环境）。

---

## 6. 常见问题 / 注意事项

| 事项 | 说明 |
|---|---|
| 16 号默认复用 | 逐个 ship_reid 系成员：其 `pred` 已存在时不重跑；要全部重跑加 `force` |
| Git | 工程根目录含 `.git`（历史提交只读）；无指令不做新提交，删除/改动前先备份 |
| 路径统一 | 所有相对路径一律相对工程根目录解析（`scripts\repro_cfg.py::resolve`）；bat 里不写死路径 |
| 唯一事实来源 | 改成员/权重/k 只改 `configs\reproduce.yaml` 的 `ensemble.members` / `ensemble.presets` / `ensemble.active` / `k` / `topk` / `member_topk` |
| 成员定义 | `{name, ckpt, pred}`；`ckpt` 相对 `sdf_checkpoint_dir`（留空 = 不做 SDF-Net 本地推理）；`pred` 相对工程根目录。ship_reid 系成员另加 `vit_config` + `vit_ckpt`（均相对 `shipvit_dir`，由 16 号自动推理；`vit_config` 必须逐成员写，因 `split_idx` 不在 ckpt 配置回填白名单内） |
| 提交文件 | 最终 = `h:\Ship-Re-Identification\prediction.json`；`submission_rrf.json` 是同内容工作副本 |
| 冠军 vs 回滚 | `ensemble.active: final_full`（波次5 全量盲提交）为冠军；`final`（缩水数据版，本地 Final 0.6905）为回滚点，两者共享同一套代码与 bat，只切换 `active` 一行 |
| 选模 | 用 `scripts\sdfnet_ckpt_sweep.py` 按 `Final` 排序；**严禁**按 SDF-Net 内置 val mAP / overall mAP 选模 |
| 保留门槛 | 每新增模块若 `Final` 提升 < 0.003 则删除；多成员选择必须做 fold 交叉验证（两折同时 ≥0.003 才保留） |
| 训练超参 | AdamW、`BASE_LR 1e-4`、`IMS_PER_BATCH 32`、`NUM_INSTANCE 2`、`MAX_EPOCHS 80`、每 5 epoch 存 ckpt；调度为 CosineLRScheduler（`t_initial=SOLVER.MAX_EPOCHS`，`SOLVER.STEPS` 实际未使用，延长训练只能改 `MAX_EPOCHS`） |
| 官方权重两份 | `SDF-Net.pth` 是官方 256×128 原始权重；`SDF-Net_256.pth` 是它的 256×256 衍生版（pos_embed 插值）。链路只用后者，两份均由 `12_sdfnet_setup.bat` 自动准备 |
| 官方模板 | `README.official.md` / `技术方案.official.md` / `路线方案.official.md` 为赛题官方起始模板（已过期），仅供参照 |

---

## 7. 参考文档

| 文档 | 内容 |
|---|---|
| `技术方案.md` | 任务与数据、评测协议与铁律、骨干/异构分支/RRF 设计、量化结果、CMAL 损失说明 |
| `路线方案.md` | 约束锚点、已完成阶段结论、负结果登记表、剩余风险、P0 优化项与下一步方向 |
| `configs\reproduce.yaml` | 路径与集成配方的唯一事实来源 |
| `experiments\` | exp_001~007 实验登记（含已否决探针及其清理记录） |
| `RRF集成方案.md` | RRF 的推导与可靠性说明（历史数值口径以 `reproduce.yaml` 为准） |