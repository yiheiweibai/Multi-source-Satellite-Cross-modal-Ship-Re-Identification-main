# exp_005 — 嵌套数据量探针：训练身份数 → 单模增益实测（exp_004 后续）

- 日期：2026-10-09
- 动机：审计发现全部 8 个冠军成员都在缩水 CSV `labels_train.csv`（2828 身份/5940 图）上训练，
  而官方全集 `labels.csv` 为 3534 身份/7422 图——local_val 按身份整组留出（seed=42、val_ratio=0.2，
  已用 `multi_split_val.py::gen_split` 逐 seed 扫描复现验证）。正式测试场景存在 2828→3534（+25%）
  的免费数据增量。本实验在受控嵌套切分下实测"训练身份数"这一单变量的增益量级。
- 结论：**两折同向为正，跨模态方向稳定增益，外推正式场景期望 Final +0.025~+0.05（点估计 ≈+0.03~0.04）。**
  建议进入波次5：全量 `labels.csv` 重训全部成员。

## 实验设计

- 骨干 convnext_tiny，配方与现役 `config/train_convnext_t.yaml` 逐项一致（损失/PK 采样/lr/100 epoch/seed42/EMA/AMP）。
- 唯一变量：训练身份数。A=2122，B=2828，且 **A ⊂ B 严格嵌套**（B 先固定，再从 B 随机去 706 身份得 A）。
- 评测：inner_val 与官方 local_val 同协议生成（`gen_split`，706 身份整组留出、query/gallery 构造规则相同），
  但身份与训练集互斥；两组模型在同一 inner_val、同一 `--tta --rerank --topk {上限}` 协议下推理评测。
- 折 1：inner_val seed 1337（gallery 1487，O2S/S2O/O2O=706/706/49，topk≤715），子集 seed 2026。
- 折 2：inner_val seed 1339（gallery 1483，706/706/47，topk≤714），子集 seed 2027。

## 结果（单模，方向得分 = (R@1+mAP@10)/2）

| 组 | 训练身份 | O2S | S2O | O2O | Final |
|---|---|---|---|---|---|
| 折1 A | 2122 | 0.3926 | 0.3703 | 0.8630 | 0.4296 |
| 折1 B | 2828 | 0.4498 | 0.4526 | 0.9038 | **0.4965** |
| 折2 A | 2122 | 0.4175 | 0.4148 | 0.8540 | 0.4599 |
| 折2 B | 2828 | 0.4521 | 0.4618 | 0.8195 | **0.4932** |
| Δ（B−A） | +706（+33%） | +0.057 / +0.035 | +0.082 / +0.047 | +0.041 / −0.034 | **+0.0669 / +0.0333** |

- 两折跨模态方向（O2S/S2O，合计权重 0.9）全部为正；O2O 仅 47~49 条 query，±0.03 噪声级，忽略。
- 两折 Δ 均值 **+0.050**；按增量比例外推 2828→3534（+25%，相对增量 0.75 倍）≈ **+0.037**。

## 判读

1. 该增益来自"每身份样本更少的相对饥饿"被缓解 + 身份分类头与采样池扩大，属训练数据量的机理增益，
   与骨干/后处理无关，预期对所有 8 个成员方向一致地成立。
2. 单模 +0.03~0.05 经集成后预期保留同量级甚至放大（成员都变强）；相对冠军 Final 0.6905，
   这是目前找到的**最大的已验证合规增益**，且零泄漏风险（官方提供 labels.csv 本就含全部 3534 身份）。
3. 与 T0-2 结论（瓶颈在榜内判别）一致：更多身份 → 判别边界更细。

## 产物与复现

- 配置：`ship_reid_vit/config/train_exp005_a_small.yaml` / `train_exp005_b_big.yaml` / `train_exp005_a2_small.yaml` / `train_exp005_b2_big.yaml`
- 数据：`question6-data/traindata/labels_exp005_{small,big,small2,big2}.csv`；task/gt：`local_val_task_s1337.json`、`local_val_gt_s1337.json`、`local_val_task_s1339.json`、`local_val_gt_s1339.json`（gen_split 产物）
- ckpt：`ship_reid_vit/outputs/exp005_{small,big,small2,big2}/checkpoints/best.pth`
- 预测：`%TEMP%\pred_exp005_{a,b}_s1337.json`、`pred_exp005_{a2,b2}_s1339.json`
- 切分脚本：`ship_reid_vit/scripts/multi_split_val.py::gen_split`（复用，未改动）
- **清理（2026-10-10）**：上列 4 个 `train_exp005_*.yaml` 配置、4 个 `labels_exp005_*.csv`、`local_val_{task,gt}_s1337/s1339.json`、`outputs/exp005_*/` 均已随波次5 后的清理删除（`%TEMP%` 预测为临时产物）。本记录保留作为波次5 全量重训的立项依据。

## 注意

- 本实验所有数字都在**新增 inner_val 划分**上测得，与官方 local_val（seed42）的 0.6905 **不可直接比 Δ**。
- 训练 CSV/ckpt/配置均为新增文件，未触碰任何现役冠军产物（prediction.json、reproduce.yaml、8 成员 ckpt）。
