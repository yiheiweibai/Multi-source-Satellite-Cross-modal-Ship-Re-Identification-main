# RRF 融合方案（当前冠军方案）

> 一句话：把 **11 份**异构 ReID 预测列表（成员各暴露 top-100 排名）用**倒数排名融合（RRF）**合并（`k=10` + O2S 方向覆盖），融合基座之上再叠 **S5 互反重排**与**闭配 α 簇修正**，**官方 Public 0.862**（α+γ 版为 0.856；融合基座官方锚点：波次6 0.763125）。全量成员（3534 身份）已见全部身份 → 融合整体**无无泄漏本地验证**（盲提交）；唯一量化证据来自「缩水成员同构子集」的无泄漏消融（8 成员 Final **0.6905**，fold0 0.7046 / fold1 0.6764，见 §4.0）。

---

## 1. 为什么用 RRF

单模型已经触顶：E0（SDF-Net ep80）位次分布显示 rank1 = 718 / 1450（top-1 = 0.495），但 **top-10 命中率已达 0.879**，即约 21.9% 的 query「正确答案就在候选列表里、只是排在第 2/3 位」。这类误差不需要新模型，只需要**重排**。

实测后被否决的路线（均 < 门槛或负收益）：
- 零成本后处理：k-reciprocal rerank 0.5785、QE 0.5847、rerank+QE 0.5815、TTA 0.5893
- P2 难负样本采样：0.4985（−0.0903）
- P6 HIMO 质量门控重排：按真实池比例折算净收益 ≈ 0
- P4 SAR 去噪：Stage A 零样本全负（lee −0.1490 / blur −0.1320 / stretch −0.0184 / clahe −0.0126 / gamma −0.0033）
- 延长训练（E0 ep80 → 160）：最优 ep155 = 0.5416（−0.0472），根因是续训重开 cosine 周期把已收敛权重重新加热

RRF 是唯一显著、且**两折交叉验证同向为正**的增益来源。

## 2. 融合原理

对每个 query，把各成员预测列表中的候选按排名折算成分数并累加：

```
score(c) = Σ_m  w_m / (k + rank_m(c))        # rank 从 1 开始；未出现在该成员列表中的候选不计分
```

最终按 score 降序取 top-10。特点：

- **只使用排名，不使用相似度量纲** → 可安全混合不同成员/协议的排名（当前冠军链路统一纯特征 top-100 排名）
- 不同模型的特征空间无需对齐，也无需归一化
- 需要调的超参只有四个：成员集、各成员权重、`k`、`member_topk`（各成员暴露的排名深度）

一个关键的数学行为：候选得分随其排名衰减 —— `w/(k+rank)`。当各成员只暴露 top-10 且 `k=10` 时，「只出现在某成员 top-10」的候选得分被压得很低，融合结果的 top-10 恒为锚点 top-10 的重排：**能修 R@1 与 mAP@10，但抬不高召回上限**（这是最初 0.5888→0.6346 的机制）。把成员排名深度提到 `member_topk=100` 并把 `k` 降到 3 之后，衰减被显著放缓：一个在多个列表里都排到第 20~60 位、被多数成员一致认可的候选，其累加分可以超过「只在锚点排到第 90 位」的候选 —— **融合因此第一次能真正抬高召回上限**，这正是 0.6346→0.6465 那段增益的来源。

## 3. 最终配方

当前 active 预设为 `final_full_b75_cm_dino_nocnxs_dir`（**11 成员**，全部为官方全量 `labels.csv` 重训版），唯一事实来源是 `configs/reproduce.yaml`：

| # | 成员 | 权重 | 预测来源 | 后处理协议 |
|---|---|---|---|---|
| 1 | `sdfnet_mos75_full`（锚点） | 1.0（**O2S 1.5**） | `logs/SDF-Net-mos-full/transformer_75.pth` → `sims/test_full/top100_mos75_full.json` | 纯特征 top-100 排名 |
| 2 | `sdfnet_ep45_full` | 1.0 | `logs/SDF-Net-full/transformer_45.pth` → `sims/test_full/top100_ep45_full.json` | 纯特征 top-100 排名 |
| 3 | `sdfnet_ep25_full` | 1.0 | `logs/SDF-Net-full/transformer_25.pth` → `sims/test_full/top100_ep25_full.json` | 纯特征 top-100 排名 |
| 4 | `ship_reid_vit_full` | 1.0（**O2S 0.5**） | `config/train_vit_full.yaml` + `outputs/full_vit/checkpoints/epoch_100.pth` → `sims/test_full/top100_shipvit_full.json` | 纯特征 top-100 排名 |
| 5 | `ship_reid_convnext_t_full` | 0.5（**O2S 0.25**） | `config/train_convnext_t_full.yaml` + `outputs/full_convnext_t/checkpoints/epoch_080.pth` → `sims/test_full/top100_convnext_t_ep080_full.json` | 纯特征 top-100 排名 |
| 6 | `ship_reid_convnext_t_strong_cm_full`（波次7） | 1.0 | `config/train_convnext_t_strong_cm_full.yaml` + `outputs/full_convnext_t_strong_cm/checkpoints/epoch_080.pth` → `sims/test_full/top100_convnext_t_strong_cm_full.json` | 纯特征 top-100 排名 |
| 7 | `ship_reid_convnext_s_pure_full`（波次4） | 0.75 | `config/train_convnext_s_pure_full.yaml` + `outputs/full_convnext_s_pure/checkpoints/epoch_040.pth` → `sims/test_full/top100_convnext_s_pure_ep040_full.json` | 纯特征 top-100 排名 |
| 8 | `ship_reid_swin_s_v2_full`（波次4） | 1.0 | `config/train_swin_s_v2_full.yaml` + `outputs/full_swin_s_v2/checkpoints/epoch_060.pth` → `sims/test_full/top100_swin_s_v2_ep060_full.json` | 纯特征 top-100 排名 |
| 9 | `sdfnet_base75`（波次6，队友互补成员） | 1.0 | 队友 ab75 配方 `base75_best.pth`（**本机只有排名，权重在队友 D 盘**）→ `sims/test/top100_base75.json` | 纯特征 top-100 排名，无稠密 sim |
| 10 | `ship_reid_dinov2_b_ep040`（波次8） | 1.0 | `config/train_dinov2_b.yaml` + `outputs/screen_dinov2_b/checkpoints/epoch_040.pth` → `sims/test_full/top100_dinov2_b_ep040_v2.json` | 纯特征 top-100 排名（tta+rerank 协议产物） |
| 11 | `ship_reid_dinov2_b_ep060`（波次8） | 1.0 | 同上 ep060 → `sims/test_full/top100_dinov2_b_ep060_v2.json` | 同上 |

**基础权重 `1 1 1 1 0.5 1 0.75 1 1 1 1`，`k = 10`、`member_topk = 100`（各成员暴露给融合的排名深度）、输出 `topk = 10`。** O2S 方向覆盖（`per_direction.O2S`）：`sdfnet_mos75_full=1.5 / ship_reid_vit_full=0.5 / ship_reid_convnext_t_full=0.25`；S2O / O2O 沿用基础权重。推理超参：SDF-Net 侧 config `configs/SDF-Net.yml`（256×256），SDF-Net 侧与 ship_reid 侧均不加 `--tta/--rerank/--qe`（推理端增强已取消，统一纯特征 top-100 排名），`--batch_size 16`。

> **成员构成演进**：波次5 全量重训（8 成员）→ 波次6 +`base75`（9）→ 波次7 +强配方 ConvNeXt-T（10）→ 波次8 +DINOv2-B ×2（12）→ 波次9 剪去 `ship_reid_convnext_s`（11）→ 波次10 叠加 O2S 方向覆盖（当前 active）。
> **波次12【官方判负】**：曾把 dino 两成员换成全量重训版，官方分 0.758 → **0.741**，已回滚。教训：全量基座上任何「已见全部身份」的新成员/新权重，在官方分出来之前一律视为未验证。
> **冠军链末段（RRF 基座之上）**：S5 互反重排（`scripts/s5_rebuild.py`，`score = RRF_base + 0.5·R2`）→ 闭配 α 簇修正（`scripts/pair_match.py` + `pair_verify.py` + `pair_anchors.json`）→ 官方 **Public 0.862**。RRF 基座本身的官方锚点为波次6 **0.763125**。

> 下表（§4）保留的是**缩水口径 8 成员同构子集**的无泄漏消融记录：这是融合增益唯一的可验证证据。演进过程：深 top-K（`member_topk=100`、`k` 从 10 降到 3）把本地验证 0.6346 → 0.6465（+0.0119）；波次 2 加异构骨干 `convnext_t` → **0.6735**；波次 3 加小权重 `ship_reid_convnext_s`（0.5）→ **0.6800**；波次 4 加两个强配方成员（0.75/1.0）→ **0.6905**。

### 3.1 成员构成说明

> 本节描述的是**缩水口径**（2828 身份）成员池的构成与筛选记录，对应 active 里各全量成员的同配方前身；当前 11 成员全量配方见上表。

- `sdfnet_mos75` 为**锚点**：CMAL 对齐损失重训出的最优 checkpoint，单模型 0.6118 > E0 0.5888，且是与 E0 不同损失训出的异构成员。
- `sdfnet_ep45` / `sdfnet_ep25` 来自 E0 同一次训练的不同收敛阶段，与锚点天然形成多样性。
- `ship_reid_vit`（ViT-Base 双分支，单模型 0.4834）与 `ship_reid_convnext_t`（ConvNeXt-Tiny 双分支 ep80，单模型 0.5186）提供**跨架构多样性**：单看分数都低于 SDF-Net，但作为异构成员价值高。
- 波次 2 筛选记录：同批训练的 Swin-Tiny 叠加后 Final 反降 0.0048（0.6735→0.6687）、ResNet50 单模仅约 0.26 且两折全为负，均按铁律（提升 < 0.003 即删）淘汰。
- 波次 3：`ship_reid_convnext_s`（ConvNeXt-Small ep60，单模型 0.5089）以**小权重 0.5** 入集，+0.0065，O2S +0.0073 / S2O +0.0070 同向上涨；同批的 ConvNeXt-Base / Swin-Small（弱配方 w_cm=0.3）40 组组合全部未过门槛，按铁律淘汰。
- 波次 4（**组合搜索 > 单成员搜索**）：两个「强配方」成员——`ship_reid_convnext_s_pure`（ConvNeXt-Small ep40，跨模态损失 w_cm 0.3→1.0 且方向对称，单模型 0.5472）与 `ship_reid_swin_s_v2`（Swin-Small ep60，同强配方，单模型 0.5547）——**各自单独入集都卡门槛 2**（k 扫描出现负增益，swin_s_v2 在 k=8~18 连续 11 个 k 的 fold0 为负），但二者同时以 **0.75 / 1.0** 权入集后**互相修补 k 空洞**：全 k 为正（min +0.0047）、三种折分割（parity / 前后半 / 相位4）G1 全部双折 ≥ +0.003、权重四邻域全部回落（内部极值，非刀锋）。同路线的 S2O 非对称方向加权（cm_dir_w_so=2.0/3.0）共 120 组合 0 通过，证明增益来自强配方本身而非方向加权。

## 4. 本地验证结果

### 4.0 缩水口径里程碑（8 成员、k=3、member_topk=100；无泄漏 local_val）

> 下表是**融合增益唯一的无泄漏量化证据**：全部成员训练在缩水 `labels_train.csv`（2828 身份），在独立 `local_val` 上评测。当前 active 的 11 个成员全部为 `labels.csv`（3534 身份）全量重训版 → **无无泄漏本地验证**，此表仅作为「RRF 在缩水同构子集上确实有效」的旁证。

| 方案 | Final | fold0 | fold1 | O2S | S2O | O2O |
|---|---|---|---|---|---|---|
| 波次2 5 成员 | 0.6735 | 0.6887 | 0.6583 | — | — | — |
| 波次3 6 成员（+convnext_s 0.5） | 0.6800 | 0.6972 | 0.6628 | 0.6605 | 0.6337 | 0.9759 |
| **波次4 8 成员（缩水口径里程碑）** | **0.6905** | **0.7046** | **0.6764** | **0.6684** | **0.6495** | **0.9741** |
| 相对波次3 的 Δ | **+0.0105** | +0.0074 | +0.0136 | +0.0080 | **+0.0158** | −0.0018 |

当前 8 成员方向明细：O2S R@1 0.6119 / mAP@10 0.7250；S2O R@1 0.5907 / mAP@10 0.7084。增益主体来自短板方向 S2O，O2O 微降 0.0018（仅 38 条、权重 0.10，不影响 Final 判定）。
k 扫描（相对 6 成员基线）6 个 k 全部两折为正：k=1 +0.0096/+0.0095、k=3 +0.0074/+0.0136、k=5 +0.0121/+0.0139、k=10 +0.0047/+0.0151、k=20 +0.0082/+0.0163、k=40 +0.0176/+0.0122（min +0.0047）。

> **口径提示**：§4.1~§4.3 与 §5 的表格是**旧口径的历史记录**（各成员只暴露 top-10、`k=10`、锚点为 E0 ep80，当时采用的 Final 为 0.6270），保留以说明 RRF 的早期机制与选择偏差排查。本节 §4.0 是**缩水口径 8 成员**的里程碑（`member_topk=100`、`k=3`、Final **0.6905**）；当前 active 的 11 成员全量配方见 §3，其 `local_val` 为 **N/A（盲）**，官方 Public 0.862。一切以 `configs/reproduce.yaml` 为准。

评测集：`local_val_task.json`（1450 query / 1482 gallery，O2S 706 / S2O 706 / O2O 38）。
fold 切分：按 `query_type` 内部交替 `folds[i % 2]` → 725 / 725。

### 4.1 成员集对比（k=10）

| 成员集 | 全量 | fold0 | fold1 | fold0 相对锚点 |
|---|---|---|---|---|
| A（锚点，单模型） | 0.5888 | 0.5757 | 0.6019 | — |
| A + ship_reid_vit | 0.6069 | 0.5827 | 0.6311 | +0.0070 |
| A + sv + ep45 | 0.6190 | 0.5969 | 0.6411 | +0.0212 |
| **A + sv + ep45 + ep25（采用）** | **0.6270** | **0.6043** | **0.6497** | **+0.0286** |
| A + sv + ep45 + ep25 + ep50 | 0.6243 | 0.6023 | 0.6464 | +0.0266 |
| 2·A + sv + ep45 | 0.6180 | 0.5953 | 0.6407 | +0.0196 |

fold1 相对锚点：采用集 +0.0478。**两折、两个跨模态方向全部同向为正。**

### 4.2 k 的敏感性

k ∈ {10, 20, 30, 40} 下采用集的全量为 0.6270 / 0.6236 / 0.6222 / 0.6222。差异仅约 0.005，**k 不是敏感超参**，取 10。

### 4.3 方向拆解（采用集，k=10，全量）

| 方向 | R@1 | mAP@10 | 方向分 | 锚点方向分 | Δ |
|---|---|---|---|---|---|
| O2S | 0.5354 | 0.6604 | 0.5979 | 0.5582 | **+0.0397** |
| S2O | 0.5156 | 0.6424 | 0.5790 | 0.5339 | **+0.0451** |
| O2O | 0.9737 | 0.9746 | 0.9741 | 0.9734 | +0.0007 |

O2O 权重仅 0.10 且样本仅 38 条，基本饱和，不构成增益来源。

## 5. 可靠性说明（重要）

- **选择偏差已被排查**：最早的「84 候选贪心融合」in-sample 拿到 0.6387（+0.0498），但 O2O（仅 38 条）跳到 0.9956 已露破绽。fold 交叉验证后，84 个候选里**只有 7 个两折同时为正、66 个两折同时为负**；贪心挑中的 `e0_256_ep30` fold0 +0.0190 而 fold1 −0.0010，确认是噪声。**该 0.6387 不可采信，已弃用。**
- **稳健底线是无折信息的 `A + ship_reid_vit`**：全量 +0.0181，fold0 +0.0070 / fold1 +0.0292。这个数字不含任何按折挑选的成分。
- **ep45 / ep25 是用「两折同时为正」筛出来的**，所以 0.6270 含少量乐观。但二者 min 增益分别为 +0.0073 / +0.0022 且两折几乎相同，稳健性好；其余候选 min ≤ 0.0022。
- **通用教训**：多成员选择必须做 fold 交叉验证，in-sample 值通常高估 2~3 倍。

## 6. 复现步骤

**推荐走 bat 链路**：在工程根目录依次执行 `14_submit_1_env_check.bat → 15_submit_2_sdfnet_infer.bat → 16_submit_3_shipvit_infer.bat → 17_submit_4_rrf_fuse.bat → 18_submit_5_verify.bat`。成员、权重、`k` / `member_topk` / `per_direction`、各成员推理用的 config/ckpt 全部由 `configs/reproduce.yaml` 决定，不需要手写命令（分步说明见 `README.md` §4）。

> ⚠️ **注意**：`17_submit_4_rrf_fuse.bat` 会把 RRF **融合基座**（非冠军）**直接写入 `h:\Ship-Re-Identification\prediction.json`**。冠军提交（α 版）还需在其上手动叠 **S5 互反重排**（`scripts/s5_rebuild.py`）与**闭配 α 簇修正**（`scripts/pair_match.py` + `pair_verify.py`）；**只跑 14~18 得到的是基座、不是冠军**。想保留原 `prediction.json` 时，先备份或给 17 加 `--out <其他路径>`。

若只想手工跑融合（cwd 为工程根目录）：

```powershell
$R    = "H:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main"
$TEST = "H:\Ship-Re-Identification\question6-data\preliminary-round-test-data\task.json"

# 方式一：直接按 reproduce.yaml 的当前预设融合（推荐，无需手写成员列表）
& "$R\.venv-sdfnet\Scripts\python.exe" "$R\scripts\repro_cfg.py" fuse --out submission_rrf.json

# 方式二：显式给成员列表（--members 与 --weights 必须等长且顺序一一对应）
#         rrf_fuse.py 只吃排名，因此可以混用不同深度的成员列表（top-100 与 top-10 都能直接传入）。
#         ⚠ 下列命令是**缩水口径 8 成员 / k=3 的历史示例**（其中 top100_convnext_s_ep060 成员已被波次9 剪枝）；
#           当前 active 的 11 成员 + `k=10` + O2S 方向覆盖请直接用「方式一」（repro_cfg.py fuse）。
& "$R\.venv-sdfnet\Scripts\python.exe" "$R\scripts\rrf_fuse.py" `
    --members sims\test\top100_mos75.json sims\test\top100_ep45.json sims\test\top100_ep25.json `
              sims\test\top100_shipvit.json sims\test\top100_convnext_t_ep080.json `
              sims\test\top100_convnext_s_ep060.json sims\test\top100_convnext_s_pure_ep040.json `
              sims\test\top100_swin_s_v2_ep060.json `
    --weights 1 1 1 1 1 0.5 0.75 1 --k 3 `
    --task $TEST `
    --out submission_rrf.json
```

成员预测由 15 号（SDF-Net 侧，`sdfnet_inference.py --topk 100`）与 16 号（ship_reid 侧，`inference.py --topk 100`，纯特征推理、无 TTA/rerank）产出：全量版落在 `sims\test_full\top100_*_full.json`，队友 `sdfnet_base75` 落在 `sims\test\top100_base75.json`。

`rrf_fuse.py` 内置校验：query 集合与 task 一致、top-10 长度、候选去重、候选全部属于 gallery。

本地验证口径复现：把 `--members` 换成各成员的**本地验证**预测（如 `ship_reid_vit/outputs/local_val_prediction.json`，或由筛选链路 `%TEMP%\t3_infer.ps1` + `t3_screen.py` 导出的本地验证榜），`--task` 换成 `local_val_task.json`，再用 `evaluate.py --submission --prediction ... --task ... --gt local_val_gt.json` 评分。

## 7. 产物清单

| 路径 | 说明 |
|---|---|
| `H:\Ship-Re-Identification\prediction.json` | **最终提交文件**（**α 版**，267 query / top-10，sha256 `3A9976D7…`，官方 **Public 0.862**）——由 RRF 基座叠 S5 + 闭配 α 得到；**注意 17 号会直接把基座写到此路径** |
| `submission_rrf.json`（工程根目录） | RRF 融合基座的**历史工作副本**（某次以 `--out submission_rrf.json` 生成；17 号 bat 默认写的是 `prediction.json`） |
| `scripts/rrf_fuse.py` | 融合脚本（通用，参数化成员/权重/k，内置平台级格式校验） |
| `scripts/repro_cfg.py` | 读取 `configs/reproduce.yaml`，导出环境变量 / 校验输入 / 按预设融合 |
| `scripts/sdfnet_inference.py` | SDF-Net 成员推理 + 后处理（`--topk` 控制导出的候选数，含 `--save_sim` 导出相似度矩阵） |
| `scripts/check_prediction.py` | 独立提交格式校验（不用 `assert`，避免 `-O` 剥离） |
| `sims/test_full/top100_{mos75,ep45,ep25}_full.json` | SDF-Net 侧全量成员预测（15 号产出，各 100 条排名） |
| `sims/test_full/top100_{shipvit,convnext_t_ep080,convnext_t_strong_cm,convnext_s_pure_ep040,swin_s_v2_ep060,dinov2_b_ep040_v2,dinov2_b_ep060_v2}_full.json` | ship_reid 侧全量成员预测（16 号产出，各 100 条排名） |
| `sims/test/top100_base75.json` | 队友 `sdfnet_base75` 成员预测（波次6 并入，本机无稠密 sim） |
| `logs/SDF-Net-full/`、`logs/SDF-Net-mos-full/` | 3 个 SDF-Net 全量成员 checkpoint（ep45 / ep25 / mos75） |
| `ship_reid_vit/outputs/full_*/checkpoints/epoch_*.pth`、`outputs/screen_dinov2_b/checkpoints/epoch_{040,060}.pth` | 7 个 ship_reid 系成员权重（由 `config/train_*_full.yaml` / `train_dinov2_b.yaml` 驱动推理） |
| `scripts/s5_rebuild.py` | 冠军链末段：S5 互反重排（`score = RRF_base + 0.5·R2`，`--fidelity` 可 267/267 逐位复现） |
| `scripts/pair_match.py`、`scripts/pair_verify.py`、`outputs/pair_diag.csv` | 冠军链末段：闭配 α 簇修正与其诊断产物 |
| `H:\Ship-Re-Identification\prediction_s5.json`、`prediction_closedpair.json`、`prediction_probe_{alpha,wbase,wcons}.json`、`prediction_ag_bak.json` | 冠军链末段中间产物（S5 输出 / 闭配版本 / 闭配 α 权重探针 / α+γ 备份） |

> **2026-10-18 清理**：已删除波次 2~4 的缩水版权重目录（`ship_reid_vit/outputs/checkpoints`、`outputs/screen_convnext_t`、`outputs/screen_convnext_s`、`outputs/screen_convnext_s_pure`、`outputs/screen_swin_s_v2`）与 `logs/SDF-Net-finetune`、`logs/SDF-Net-mos` → `reproduce.yaml` 中 `final` / `final_full` / `final_full_b75` 等**历史预设不再可重跑**，现役只保留全量版权重（`reproduce.yaml` 本身不改，保留历史记录）。
> `ship_reid_vit/outputs/prediction.json` 是 `05_inference.bat` 的旧产物路径，**已不再作为融合输入**；同一成员的融合输入现为 `sims/test_full/top100_shipvit_full.json`（深 top-K 口径）。

## 8. 测试集终检

`preliminary-round-test-data/task.json`：267 query / 1800 gallery（O2S 120 / S2O 120 / O2O 27）。

- query 覆盖 267/267，候选全部落在 gallery 内，无重复，长度均为 10
- 候选模态错配 **0**（O2S→sar、S2O/O2O→optical）
- query 自身图像出现在 top-10 的条数 **0**
- 7 个 ship_reid 系全量成员预测（`top100_shipvit_full` / `top100_convnext_t_ep080_full` / `top100_convnext_t_strong_cm_full` / `top100_convnext_s_pure_ep040_full` / `top100_swin_s_v2_ep060_full` / `top100_dinov2_b_ep040_v2` / `top100_dinov2_b_ep060_v2`）逐项复核：query 全覆盖、ID 全合法、模态无错配、无 query 自匹配、成员列表长度均为 `member_topk=100`

## 9. 已知局限

- **全量成员基座无无泄漏验证**（最重要）：11 个成员全部在 `labels.csv`（3534 身份）重训，已见全部身份 → 融合与后续 S5 / 闭配 α 的任何改动都无法做无泄漏本地验证；§4.0 的 0.6905 只是缩水同构子集的旁证。**波次12 的官方判负（0.758→0.741）正是该风险的真实例证**——新增成员/权重一律在官方分出来前视为未验证。
- **S5 与闭配 α 属保守后处理**：S5（`score = RRF_base + 0.5·R2`）对 top-1 的翻转仅个位数、平均交集 9.63/10，属小步微调；闭配 α 的增益未在独立验证折上量化。官方分对照：α **0.862**（当前提交）> RRF 基座 0.758 > 波次12 0.741；α+γ 版为 0.856。
- **深 top-K 首次抬高了召回上限**：`member_topk=100 / k=3` 后，融合不再局限于锚点 top-10 的重排（对比见 §2）。但成员深度也只有 100，超过该深度的正确答案仍无法进入输出，本地验证的召回上限因此并非无界。
- O2O 仅 38 条（测试集 27 条），该方向分数噪声大，不适合作为选模依据。
- 成员集与 `k` / `member_topk` 均在**缩水口径**本地验证上选定，若测试集分布与验证集差异较大，增益可能缩水。
- ship_reid 系成员若重训或换骨干，必须重新做三重门槛（两折 ≥ +0.003、k 扫描不翻负、人工复核）再决定是否替换；**新增成员筛选不能只测单独追加/替换**——波次4 证明两个各自卡门槛 2 的候选可联合入集互相修补 k 空洞，但该判定必须用三种以上折分割 + 权重邻域 + 换成员对照防多重检验偏差。