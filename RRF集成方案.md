# RRF 融合方案（当前冠军方案）

> 一句话：把 4 份异构 ReID 预测列表用**倒数排名融合（RRF）**合并，本地验证 Final 从 0.5888 提到 **0.6270**，且两个跨模态方向同向改善。

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

- **只使用排名，不使用相似度量纲** → 可安全混合「纯特征 top-10」与「TTA+rerank 后的 top-10」等不同后处理协议
- 不同模型的特征空间无需对齐，也无需归一化
- 需要调的超参只有两个：成员集与 k

一个有用的数学性质：若候选只出现在 A 的 top-10，其得分 ≥ w_A/(k+10)。要让「只出现在 B 的 top-10」的候选翻盘，需满足 `w_B > 0.871 · w_A`（k=10 时）。因此当 `w_B ≤ 0.87 · w_A` 时，**融合结果的 top-10 恒为 A 的 top-10 的重排**——不可能抬高召回上限，但能修 R@1 与 mAP@10。这正是本方案有效的机制。

## 3. 最终配方

| 成员 | 权重 | 权重文件 / 预测来源 | 后处理协议 |
|---|---|---|---|
| SDF-Net ep80（锚点 A） | 1.0 | `logs/SDF-Net-finetune/transformer_80.pth` | 纯特征 top-10 |
| SDF-Net ep45 | 1.0 | `logs/SDF-Net-finetune/transformer_45.pth` | 纯特征 top-10 |
| SDF-Net ep25 | 1.0 | `logs/SDF-Net-finetune/transformer_25.pth` | 纯特征 top-10 |
| ship_reid_vit | 1.0 | `ship_reid_vit/outputs/prediction.json` | TTA + rerank top-10 |

**k = 10，等权。** 推理超参：`--batch_size 16`，config `configs/SDF-Net.yml`（256×256），不加 `--tta/--rerank/--qe`。

## 4. 本地验证结果

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

```powershell
$S = "H:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\ship_reid_system"
$TEST = "H:\Ship-Re-Identification\question6-data\preliminary-round-test-data\task.json"

# 1) 三个 SDF-Net checkpoint 在测试集上推理（cwd 必须是 SDF-Net，config 里 PRETRAIN_PATH 是相对路径）
#    每个约 1~2 分钟，输出 sims/test/pred_ep{80,45,25}.json
foreach ($ep in 80,45,25) {
  & "$S\.venv-sdfnet\Scripts\python.exe" "$S\scripts\sdfnet_inference.py" `
      --config_file configs/SDF-Net.yml `
      --weight "$S\logs\SDF-Net-finetune\transformer_$ep.pth" `
      --task_json $TEST `
      --out_prediction "$S\sims\test\pred_ep$ep.json" `
      --batch_size 16
}

# 2) RRF 融合（cwd 为 ship_reid_system），输出 ship_reid_system/submission_rrf.json
& "$S\.venv-sdfnet\Scripts\python.exe" "$S\scripts\rrf_fuse.py" `
    --members sims\test\pred_ep80.json sims\test\pred_ep45.json sims\test\pred_ep25.json `
              "H:\Ship-Re-Identification\ship_reid_vit\outputs\prediction.json" `
    --weights 1 1 1 1 --k 10 `
    --task $TEST `
    --out submission_rrf.json
```

`rrf_fuse.py` 内置校验：query 集合与 task 一致、top-10 长度、候选去重、候选全部属于 gallery。

本地验证口径复现：把 `--members` 换成 `sims/sweep/e0_256/pred_ep{80,45,25}.json` 与 `ship_reid_vit/outputs/local_val_prediction.json`，`--task` 换成 `local_val_task.json`，再用 `evaluate.py --submission --prediction ... --task ... --gt local_val_gt.json` 评分。

## 7. 产物清单

| 路径 | 说明 |
|---|---|
| `H:\Ship-Re-Identification\prediction.json` | **最终提交文件**（文件名符合平台要求，267 query，top-10） |
| `ship_reid_system/submission_rrf.json` | 同一内容的工作副本（融合脚本默认输出名） |
| `ship_reid_system/scripts/rrf_fuse.py` | 融合脚本（通用，参数化成员/权重/k，内置平台级格式校验） |
| `ship_reid_system/scripts/sdfnet_inference.py` | 单模型推理 + 后处理（含 `--save_sim` 导出相似度矩阵） |
| `ship_reid_system/sims/test/pred_ep{80,45,25}.json` | 测试集成员预测 |
| `ship_reid_system/sims/sweep/e0_256/pred_ep*.json` | 本地验证成员预测（32 个 checkpoint） |
| `ship_reid_vit/outputs/prediction.json` | ship_reid_vit 测试集预测（复用，未改动） |

## 8. 测试集终检

`preliminary-round-test-data/task.json`：267 query / 1800 gallery（O2S 120 / S2O 120 / O2O 27）。

- query 覆盖 267/267，候选全部落在 gallery 内，无重复，长度均为 10
- 候选模态错配 **0**（O2S→sar、S2O/O2O→optical）
- query 自身图像出现在 top-10 的条数 **0**
- 相对锚点 A 单模型：top-1 变动 64/267（24%），top-10 集合变动 249/267，融合新引入的非 A-top10 候选共 467 个

## 9. 已知局限

- 收益机制是「在锚点 top-10 内重排」，**不抬高召回上限**（本地验证 R@1 上限仍为 0.879）。当前 O2S R@1 0.5354 距 0.879 仍有大量空间，瓶颈在召回而非排序。
- O2O 仅 38 条（测试集 27 条），该方向分数噪声大，不适合作为选模依据。
- 成员集与 k 均在本地验证上选定，若测试集分布与验证集差异较大，增益可能缩水（保守估计仍应保住 `A + ship_reid_vit` 的 +0.018）。
- `ship_reid_vit` 侧若重训/换骨干，必须重新做一轮 fold 交叉验证再决定是否替换成员。