# exp_007 — 外部 SAR 自监督预训练（OSSDD 域自适应 DAPT）探针

## 目的与判据
判断「把外部 SAR 舰船切片集以自监督方式做成初始化」能否提升 HOSS 跨模态舰船 ReID。
判据：单骨干 convnext_t，local_val 上对比 ImageNet-init 基线，**O2S/S2O 四格方向全正且 Δ≥+0.003** 才算过。

## 协议（唯一变量 = 初始化来源）
- **SSL 语料**：OSSDD（`sylviaHoch/OpenSARShip-Ship-Detection-Dataset`），30 个 train 分片（14.71 GB，3,570 样本 / 9,987 框）
  → 按 aabb 船中心自适应裁剪 → **19,974 张 256×256 灰度切片**（`H:\ship_reid_ext\ossdd_corpus\img`）
  渲染：clip(>0,≤20000) → 10·log10 → 逐图 p1/p99.5 拉伸 → uint8
- **SSL**：SimCLR DAPT，backbone `convnext_tiny`（timm 默认 `in12k_ft_in1k`），batch 128，lr 2e-4，wd 0.05，τ=0.2，aug=RandomResizedCrop/翻转/ColorJitter/灰度/高斯模糊
  计划 60 epoch；按指示「拿到 1 个 ckpt 即停」，实际训到 **ep10**（loss 2.64→**1.4484**）产出 ckpt
- **下游**：`config/train_convnext_t_ssl.yaml`，由冠军 `train_convnext_t.yaml` 复制，**仅改初始化**（`pretrained:false` + `pretrained_path`）与 `project_name/output_dir`；lr/PK/损失/归一化/数据(streaked `labels_train.csv`)/轮数逐项一致
- **初始化装载校验**：`optical_encoder` 与 `sar_encoder` 的 stem 卷积权重与 SSL ckpt `stem.0.weight` 的 max|diff| = **0.000e+00**（已确认载入，非随机初始化）
- **评测**：`inference.py --local_val_task --topk 100 --tta --rerank` → `eval_localval.py`（与 t3_screen.py 同 Fuser/fold 口径）

## 结果（local_val，1450 query / 1482 gallery）

| 模型 | Final | fold0 | fold1 | O2S | S2O | O2O |
|---|---|---|---|---|---|---|
| baseline convnext_t **ep080**（ImageNet-init） | 0.5186 | 0.5308 | 0.5063 | 0.4657 | 0.4763 | 0.9463 |
| baseline convnext_t **ep060**（ImageNet-init） | 0.5095 | 0.5306 | 0.4884 | 0.4570 | 0.4645 | 0.9483 |
| **SSL ep060**（OSSDD DAPT-init） | **0.4651** | 0.4531 | 0.4772 | 0.4252 | 0.4168 | 0.8618 |

同 epoch 对比（SSL ep060 vs baseline ep060）：
Final **−0.0444**；O2S −0.0318；S2O −0.0477；O2O −0.0865；fold0 −0.0775；fold1 −0.0112

## 结论：判负（REJECT）
四格方向全负，幅度远超 0.003 门槛——**不是「信号太弱」，而是显著劣化**。
机制推测（按可能性排序）：
1. **SAR-only 语料污染了 optical 分支**：下游是双分支（浅层 optical/sar 各一份、初始权重相同），把 SAR 偏向权重同时灌给 optical 分支 → 直接损伤光学表征，与 **O2O −0.0865** 高度吻合。
2. **域差异**：OSSDD 是 Sentinel-1 GRD 检测切片（log 拉伸、含海杂波/海岸），与 HOSS「纯黑背景 + 稀疏散射点船体」统计不同，DAPT 把特征推向源域。
3. **归一化不一致**：SSL 用 ImageNet norm，下游 SAR 用 `sar_mean/std = 0.0242/0.1214`。
4. SSL 仅 10 epoch（loss 1.4484）——但效应是强负，不支持「训练不足」作为主因。

## 产物
- SSL ckpt：`H:\ship_reid_ext\ssl_convnext_tiny_ossdd_ep010.pth`（111 MB）
- 语料：`H:\ship_reid_ext\ossdd_corpus\img`（19,974 png）｜分片：`H:\ship_reid_ext\ossdd\webdataset\train\*.tar`（30 个）
- 下游权重：`ship_reid_vit/outputs/screen_convnext_t_ssl/checkpoints/epoch_{020,040,060}.pth`
- 预测：`sims/t1/top100_convnext_t_ssl_ep060.json`、`sims/t1/top100_convnext_t_ep060.json`
- 配置：`ship_reid_vit/config/train_convnext_t_ssl.yaml`
- 脚本（项目目录外）：`H:\ship_reid_ext\ssl_pretrain.py`、`eval_localval.py`、`prepare_corpus.py`

> **清理（2026-10-10）**：项目外 `H:\ship_reid_ext`（SSL ckpt / 语料 19,974 png / 30 个 OSSDD 分片 / 脚本）与下游 `outputs/screen_convnext_t_ssl/`、`config/train_convnext_t_ssl.yaml`、`sims/t1/top100_convnext_t*_ep060.json` 已全部删除。本记录保留作为该线判负的依据。

## 后续（若要继续该线，需改设计而非加轮数）
- **非对称初始化**：仅把 SSL 权重注入 `sar_encoder`，`optical_encoder` 保留 ImageNet——直击机制 1
- SSL 语料混入光学/多域数据，或改用模态无关的掩码重建（MAE）
- 延长 SSL 至 100+ epoch 并做 ckpt 扫描（成本高，优先级低于上两条）

## 合规
赛题规则第 127 行允许外部数据/预训练模型（须在复现材料完整披露）。OSSDD 为公开数据集（标注 CC-BY-NC-SA-4.0 + Copernicus Sentinel 影像条款）。