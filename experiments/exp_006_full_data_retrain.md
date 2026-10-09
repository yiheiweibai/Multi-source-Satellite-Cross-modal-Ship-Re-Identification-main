# exp_006 — 波次5：全量数据（labels.csv 3534 身份）重训 8 成员并盲提交

- 日期：2026-10-09
- 动机：exp_005 已受控证明「训练身份数」是当前最大的合规增益源（嵌套双折 2122→2828
  单模 Final +0.0669 / +0.0333，两折同向、O2S/S2O 四格全正）。本实验把该结论落到生产：
  把冠军 8 成员的训练数据从缩水 CSV `labels_train.csv`（2828 身份/5940 图）换成官方全集
  `labels.csv`（3534 身份/7422 图，+25%）重训，其余全部沿用冠军设定。
- 结论：**8 成员全部重训完成并盲提交**（`H:\Ship-Re-Identification\prediction.json`，
  sha256 `F68B90EB…`，格式终检 PASS）。因全量成员已见全部 3534 身份，**不存在无泄漏本地验证**，
  本批属盲提交；期望 Final +0.02~+0.04（exp_005 外推），未经本地证实。

## 设计（唯一变量 = 训练身份数）

| 项 | 现役冠军成员 | 波次5 成员 |
|---|---|---|
| 训练 CSV | `labels_train.csv`（2828 身份/5940 图） | `labels.csv`（3534 身份/7422 图） |
| 骨干 / 配方 / PK 采样 / lr / seed | — | **逐项沿用，未改** |
| 取点 epoch | vit ep100 / cnx_t ep80 / cnx_s ep60 / pure ep40 / swin_v2 ep60；SDF-Net ep25、ep45、ep75 | **同** |
| 集成 | k=3、member_topk=100、权重 [1,1,1,1,1,0.5,0.75,1] | **同** |

- SDF-Net 侧：训练走目录 glob（`datasets/hoss.py`），缩水版实为替换 `data/HOSS/bounding_box_train`。
  新建 `SDF-Net/data/full/HOSS/bounding_box_train`（labels.csv 全量 7422 图的 hardlink，3829 RGB +
  3593 SAR，3534 身份），query / bounding_box_test 用 junction 复用现役目录（内置 EVAL 仅作日志）。
  配置：`configs/SDF-Net-full.yml`（finetune 配方）、`configs/SDF-Net-mos-full.yml`（+CMAL_LOSS_WEIGHT=1.0）。
- ship_reid_vit 侧：`config/train_{vit,convnext_t,convnext_s,convnext_s_pure,swin_s_v2}_full.yaml`，
  仅改 `train_labels_csv` → labels.csv、输出目录换名；`save_every` 5→20（纯存盘频率，磁盘预算）。

## 训练实测

| 成员 | 框架 | 耗时 | 保留 ckpt | 备注 |
|---|---|---|---|---|
| ship_reid_vit | shipvit | 2965s | `outputs/full_vit/checkpoints/epoch_100.pth` | |
| convnext_t | shipvit | 1461s | `outputs/full_convnext_t/checkpoints/epoch_080.pth` | |
| convnext_s | shipvit | 2109s | `outputs/full_convnext_s/checkpoints/epoch_060.pth` | |
| convnext_s_pure | shipvit | 2106s | `outputs/full_convnext_s_pure/checkpoints/epoch_040.pth` | |
| swin_s_v2 | shipvit | 2534s | `outputs/full_swin_s_v2/checkpoints/epoch_060.pth` | |
| sdfnet_mos75 | SDF-Net | 5633s | `logs/SDF-Net-mos-full/transformer_75.pth` | CMAL 项真实非零（0.026），全量多图使方差项不再退化 |
| sdfnet_ep45 | SDF-Net | 5128s | `logs/SDF-Net-full/transformer_45.pth` | |
| sdfnet_ep25 | SDF-Net | 5128s | `logs/SDF-Net-full/transformer_25.pth` | |

合计约 6.1h（单卡 RTX 4070 Ti SUPER，串行；`%TEMP%\w5_train_all.py` 驱动，含非冠军快照清理）。

## 健康度检查（盲提交下唯一可用的证据）

1. **逐成员差异**（新 vs 现役同名成员，267 test query，top-1 一致率 / top-10 平均交集）：
   mos75 0.693/6.68、ep45 0.674/6.13、ep25 0.648/6.59、shipvit 0.524/5.61、
   cnx_t 0.562/5.43、cnx_s 0.566/5.12、pure 0.569/6.11、swin_v2 0.663/**7.57**
   → 重训实质性改变了每个成员（既非复制也非崩坏）。
2. **整体**：candidate vs 现役冠军 top-1 一致率 **0.824**、top-10 平均交集 **7.85/10**、
   top-100 Jaccard 0.663、唯一候选数均 835、无自身匹配 → 典型「多数一致、难例分歧」形态。
3. **交叉成员一致性**：新旧两套的成员间一致性矩阵**结构完全保留**（SDF-Net 三兄弟 6.6~6.9，
   异构五成员 4.1~5.9），新成员平均一致性仅整体下移 0.04~0.22，**无成员塌陷**。
4. **格式**：`check_prediction.py` PASS（267/1800、恰好 top-10、O2S/S2O/O2O=120/120/27、模态一致）。
5. **链路自洽**：更新后的 `configs/reproduce.yaml`（`active: final_full`）重跑 15→17→18
   复现出**逐字节相同**的 prediction.json（sha256 MATCH）。

## 产物与回滚

- 提交：`H:\Ship-Re-Identification\prediction.json`（sha256 `F68B90EB…`，51267 B）
- 候选：`prediction_w5_candidate.json`（同哈希）
- 回滚点：`prediction.backup_8members_champion_0.6905.json`（上一版冠军，Final 0.6905）
- 成员预测：`sims/test_full/top100_*_full.json`（8 份）；`reproduce.yaml` 中 `final` 预设保留
  （2828 身份版 8 成员，改 `ensemble.active: final` 即可回退）
- 驱动脚本（`%TEMP%`）：`w5_train_all.py`（训练）、`w5_infer_fuse.py`（推理+融合）、
  `w5_compare.py` / `w5_consistency.py`（对比与一致性）、`w5_chain_verify.py`（链路复现校验）

## 注意

- 与 exp_005 相同，**本批数字与 0.6905 不可直接比 Δ**（验证划分不同 + 本批无泄漏验证）。
- 取点 epoch 与集成参数（k / 权重）均沿用旧冠军：因全量成员无法做无泄漏选模，这是无法回避的假设。
  若后续要重调，需要另建「封闭身份子集」训练 + 独立验证折的设计。
- 该增益属数据级机理增益，与骨干/后处理无关；下一步若继续提分，应在此全量基线上叠加
  （更多骨干 / 更强预训练 / 更强跨模态配方），而非回到缩水数据上做重排微调。