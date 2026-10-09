# experiments — 实验记录体系

> 对齐 `路线方案.md / roadmap.md` 第 2.2 节"统一记录"约定。
> 实验闭环的核心是**可追溯、可复现、可对比**：任何一次推理/评测结果都必须落到一张 CSV 里，
> 并给出相对 baseline 的 Δ，供"只保留正增益项"的决策使用。

## 记录规范（每个实验必记）

| 字段 | 说明 |
|---|---|
| `exp` | 实验编号（exp_001 / exp_002 / ...），与 README 中实验说明对应 |
| `model` | 使用的模型（如 TransOSS / 自研 CrossModal ReID） |
| 数据处理 / TTA / Rerank / QE / Cluster | 推理端开关：`×` 关闭，`✓` 开启；数据处理列写具体方案 |
| `O2S` / `S2O` / `O2O` | 三个方向的得分（evaluate.py --submission 输出，R@1 与 mAP@10 的平均） |
| `Overall` | 综合得分 |
| `ΔBaseline` | `Overall - baseline Overall`（同一验证集、同一划分才可比），正增益保留、负增益剔除 |
| `备注` | 验证集划分批次 / Public 对应关系 / 参数固定值等 |

## 单变量原则

> **每次只改变一个核心变量。** 对比两个实验时，除目标变量外的一切（验证集划分、权重、
> 评测脚本、后处理参数）必须保持一致，否则 Δ 无意义。

- 验证集：统一复用 `../赛题6-初赛/训练数据/local_val_task.json`（09 脚本生成的 20% 验证集），
  不重新划分；若必须重新划分，在 `备注` 里写明划分批次与 seed，且**不与旧批次直接比较 Δ**。
- 权重：未特别说明时统一 `logs/competition_transoss/transformer_200.pth`。
- Rerank 参数：k1=20、k2=6、lambda=0.3（脚本内固定）。
- TTA：水平翻转 feature average（路线图 7.1）。

## 判定标准

1. `ΔBaseline > 0`：本地验证集有增益，可考虑加入最终提交配置。
2. `ΔBaseline ≤ 0`：本地验证集无增益，**不因"论文里有效"而加入**（路线图第 19 节）。
3. Public 是最终裁判，本地验证集是实验裁判；Public 与本地验证集分布不同，
   **禁止**把 Public 分数当作本地 Δ 的依据直接比较。

## 已登记实验

| 文件 | 实验内容 | 状态 |
|---|---|---|
| `exp_001_transoss_base.csv` | TransOSS 纯 cosine baseline（两次划分 + Public） | 已登记 |
| `exp_002_ckpt_fusion.csv` | 10 个 checkpoint 逐点评测 + RRF(top3) 融合 | 已登记 |
| `exp_003_sar_preprocess_ab.csv` | 11 脚本六组合消融（base/preprocess/colormap/tta/rerankqe/mixed） | 模板，待 11 跑完填写 |
| `exp_004_t0_gain_probe.md` | T0 零训练四探针：CLIP 弱配方骨干 / 深 member_topk / best·last / 方向独立 k（全部 REJ，含 union top-10 召回已 98.83% 的实测） | 已登记 |
| `exp_005_nested_data_scaling.md` | 嵌套数据量探针：训练身份 2122→2828 双折实测（+0.067/+0.033 同向），外推全量重训期望 +0.03~0.04 | 已登记 |
