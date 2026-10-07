# 赛题06 多源卫星跨模态舰船重识别 · 复现指南

> **一句话**：光学（Optical）与 SAR 两种模态下同一艘舰船的跨模态身份匹配。最终方案 = **RRF 倒数排名融合**（3 个 SDF-Net checkpoint + 1 个异构 ViT 分支），本地验证 Final **0.6346**（fold0 0.6131 / fold1 0.6561），单模型冠军基线 E0 为 0.5888。

本文档面向「能在本机把最终提交复现出来」的人。技术选型理由见 `技术方案.md`，历史取舍与后续方向见 `路线方案.md`。

---

## 1. 当前状态

| 项 | 值 |
|---|---|
| 赛题 | 2026 全国大数据与计算智能挑战赛 · 赛题06「多源卫星跨模态舰船重识别」 |
| 测试集 | 267 query / 1800 gallery（O2S 120 / S2O 120 / O2O 27） |
| 选模判据 | `Final = 0.45×O2S + 0.45×S2O + 0.10×O2O`（方向分 = (R@1 + mAP@10)/2） |
| 冠军方案 | RRF 融合，`configs/reproduce.yaml` 的 `ensemble.active: final` |
| 本地验证 Final | **0.6346**（fold0 0.6131 / fold1 0.6561） |
| 单模型冠军基线 | E0 = SDF-Net ep80：0.5888 |
| 最终提交文件 | `h:\Ship-Re-Identification\prediction.json` |
| 项目 Git | **无 Git 仓库**，任何改动/删除不可逆 |

---

## 2. 目录结构（与磁盘一致）

工程根目录 = `h:\Ship-Re-Identification\Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\`（已扁平化，**没有** `ship_reid_system/` 这一层）。

```
h:\Ship-Re-Identification\
├── prediction.json                     ★ 最终提交文件（267 query, top-10）
├── question6-data\                     赛题数据（只读）
│   ├── traindata\                      labels_train.csv / local_val_task.json / local_val_gt.json
│   └── preliminary-round-test-data\    task.json（初赛测试集）
└── Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main\   ← 工程根目录
    ├── 12_sdfnet_setup.bat             环境准备（venv / 依赖 / 官方权重 / HOSS 数据）
    ├── 13_sdfnet_plan.bat              分阶段对比编排（历史遗留入口，见 §4.6）
    ├── 14_submit_1_env_check.bat       ① 环境自检
    ├── 15_submit_2_sdfnet_infer.bat    ② SDF-Net 成员推理
    ├── 16_submit_3_shipvit_infer.bat   ③ ship_reid_vit 推理
    ├── 17_submit_4_rrf_fuse.bat        ④ RRF 融合
    ├── 18_submit_5_verify.bat          ⑤ 提交格式终检
    ├── 19_mos_retrain.bat              CMAL 对齐损失重训（产物 logs\SDF-Net-mos）
    ├── configs\
    │   └── reproduce.yaml              ★ 唯一事实来源（路径 / 成员池 / 权重预设 / k）
    ├── scripts\
    │   ├── repro_cfg.py                解析 reproduce.yaml（相对路径均相对工程根目录）
    │   ├── rrf_fuse.py                 RRF 融合 + 平台级格式校验
    │   ├── sdfnet_inference.py         单模型推理 + 按 query_type 分组后处理
    │   ├── sdfnet_ckpt_sweep.py        checkpoint 选优（按 Final 排序）
    │   ├── sdfnet_prepare_data.py      赛题数据 → HOSS 布局
    │   ├── sdfnet_download_weights.py  官方权重下载
    │   ├── check_prediction.py         提交格式独立校验器
    │   └── build_local_val.py / compute_stats.py
    ├── config\  data\  models\  inference\  utils\   自研模块（保留）
    ├── evaluate.py                     赛题提交评测入口（--submission）
    ├── SDF-Net\                        主骨干（上游 cfrfree/SDF-Net + 本项目改动）
    │   ├── configs\SDF-Net.yml         E0 微调配置
    │   ├── configs\SDF-Net-mos.yml     MOS 重训配置（唯一差异：CMAL_LOSS_WEIGHT=1.0）
    │   ├── loss\cmal_loss.py           CMAL 对齐损失
    │   └── ...
    ├── ship_reid_vit\                  异构 ViT 分支（与 SDF-Net 并列）
    │   ├── 01_setup.bat … 05_inference.bat
    │   ├── config\train_vit.yaml
    │   ├── outputs\checkpoints\best.pth
    │   └── weights\vit_base_patch16_224.pth
    ├── logs\
    │   ├── SDF-Net-finetune\           E0 微调（transformer_5…80.pth） ★ 成员 ep45/ep25 来源
    │   ├── SDF-Net-mos\                MOS 重训（transformer_75.pth） ★ 锚点来源
    │   └── SDF-Net-ext160\ SDF-Net-hn\ …（已否决方案归档）
    ├── sims\
    │   ├── test\pred_mos75/45/25/80.json   测试集成员预测
    │   └── sweep\…                         本地验证选优产物
    ├── experiments\                    实验记录 CSV
    └── *.official.md                   赛题官方起始模板（只读，已过期）
```

---

## 3. 环境准备

需要**两个**独立虚拟环境，当前两者都已被删除，端到端复现前必须重建。

### 3.1 `.venv-sdfnet`（14~19 号脚本使用）

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

可用 `SDF_DATA_DIR` 覆盖数据目录。12 号结束后直接进入 §4（从 14 号开始）；13 号是历史遗留入口（见 §4.6），仅在其内部微调时才需要，16GB 显卡 OOM 时在那条链路上设 `SDF_IMS_PER_BATCH=16` 即可（该变量只被 13 号读取，final 链路的 batch size 在 yaml 里）。

> **两个权重文件的分工**：`SDF-Net.pth` 是官方原始 256×128 权重（下载产物）；**`SDF-Net_256.pth` 才是本项目 256×256 链路实际使用的那份**——`reproduce.yaml` 的 `sdfnet_official` 与 `SDF-Net.yml` / `SDF-Net-mos.yml` 的 `PRETRAIN_PATH`、`WEIGHT` 均指向它，14 号自检检查的也是它。两者都由 12 号自动准备完成，无需手工干预。

### 3.2 `ship_reid_vit\venv`（仅 16 号 ship_reid_vit 侧推理使用）

```bat
:: 进入 ship_reid_vit 目录后执行：
01_setup.bat
```

创建 `ship_reid_vit\venv`、安装 torch/torchvision（CUDA 12.4，失败回落 CPU）+ `requirements.txt`，并下载 `weights\vit_base_patch16_224.pth`。

> 注意：`ship_reid_vit` 的训练（`03_train.bat`）会重新生成 `outputs\checkpoints\best.pth`；若该权重已存在且无需重训，可跳过训练直接进入 §4。

---

## 4. bat 脚本（12~19）与 final 流程

### 4.0 全部 bat 一览（12~19）

| # | 脚本 | 定位 | 做什么 | 运行环境 | 主要输入 → 输出 |
|---|---|---|---|---|---|
| 12 | `12_sdfnet_setup.bat` | 环境准备（一次性） | 建 `.venv-sdfnet`；装 `torch 2.2.2+cu118` / `torchvision 0.17.2+cu118` + `numpy/opencv-python/Pillow/thop/timm/yacs`；克隆上游 SDF-Net（已存在且含 `train.py` 则跳过）；从 HuggingFace `Chenfree233/SDF-Net` 下载官方权重 `SDF-Net.pth`（约 333MB）；**派生 256×256 版本 `SDF-Net_256.pth`**（pos_embed 插值，final 链路真正使用的那份）；把赛题数据转成 HOSS 布局 | 系统 Python（脚本自建 venv） | `question6-data\traindata\` → `.venv-sdfnet\`、`SDF-Net\data\HOSS\{bounding_box_train,bounding_box_test,query}`、`SDF-Net\logs\SDF-Net\SDF-Net.pth` + `SDF-Net_256.pth` |
| 13 | `13_sdfnet_plan.bat` | 历史遗留入口 | 分阶段对比编排：A1/A2（官方权重 base / +rerankqe）、B1/B2（微调后 base / +rerankqe）、C（TransOSS 重测）、D1~D4（SDF-Net × TransOSS 分数/RRF 融合）。**C/D 依赖已删除的 TransOSS，本就不可运行**；A/B 仍可当微调入口 | `.venv-sdfnet` | 官方权重 / 微调权重 → `experiments\exp_005_sdfnet_plan_compare.md`（对比文档） |
| 14 | `14_submit_1_env_check.bat` | **final ①** | 解析 `reproduce.yaml` 把路径导出为环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项自检当前预设所需输入是否齐备 | `.venv-sdfnet` | `configs\reproduce.yaml` → 控制台 OK/MISS 清单；齐备退出码 0 |
| 15 | `15_submit_2_sdfnet_infer.bat` | **final ②** | 对当前预设中**带 `ckpt` 的 SDF-Net 成员**逐个在测试集推理（cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）；`SDF_N=0` 时直接跳过 | `.venv-sdfnet` | `logs\SDF-Net-mos\transformer_75.pth`、`logs\SDF-Net-finetune\transformer_45/25.pth`、`task.json` → `sims\test\pred_mos75.json` / `pred_ep45.json` / `pred_ep25.json` |
| 16 | `16_submit_3_shipvit_infer.bat` | **final ③** | 用异构分支对测试集推理（`--tta --rerank`）。**默认行为：`ship_reid_vit\outputs\prediction.json` 已存在则直接复用、不重跑**（保证结果可比）；要重跑加参数 `force` | `ship_reid_vit\venv` | `config\train_vit.yaml` + `outputs\checkpoints\best.pth` → `ship_reid_vit\outputs\prediction.json` |
| 17 | `17_submit_4_rrf_fuse.bat` | **final ④** | RRF 倒数排名融合：`score(c) = Σ_m w_m / (k + rank_m(c))`，等权、`k=10`、每 query 取 top-10；成员与权重在 Python 侧按序绑定（不经过 cmd） | `.venv-sdfnet` | 全部成员预测（`final` 预设 4 个文件） → `h:\Ship-Re-Identification\prediction.json`（工作副本 `submission_rrf.json`） |
| 18 | `18_submit_5_verify.bat` | **final ⑤** | 调 `scripts\check_prediction.py` 逐条对照平台「直接判定无效」条件核验提交文件，并回显本地验证基准 | `.venv-sdfnet` | `prediction.json` + `task.json` → PASS/FAIL 报告（基准 Final 0.6346 / fold0 0.6131 / fold1 0.6561） |
| 19 | `19_mos_retrain.bat` | 可选重训 | 用 CMAL 对齐损失重训锚点模型（`SDF-Net-mos.yml` 相对 `SDF-Net.yml` 的唯一差异是 `CMAL_LOSS_WEIGHT=1.0`），每 5 epoch 存一次 checkpoint；**从官方权重起训，不是 resume** | `.venv-sdfnet` + GPU | 官方权重 → `logs\SDF-Net-mos\transformer_*.pth`（用 `sdfnet_ckpt_sweep.py` 按 Final 选优） |

> 分工：**14→15→16→17→18** 是唯一的 final 提交链路（下文 §4.1~§4.5 逐步展开）；**12** 是一次性环境准备；**13** 是历史遗留入口；**19** 仅在需要替换锚点时才跑。

统一约定：所有步骤在**工程根目录**执行；每一步都会重新解析 `configs/reproduce.yaml`（不读取陈旧的环境变量文件）。产物链路：

```
成员预测（sims/test/pred_mos75.json / pred_ep45.json / pred_ep25.json + ship_reid_vit/outputs/prediction.json）
        └──────────────► 17 号 RRF 融合 ──► ../prediction.json（最终提交）
```

### 4.1 `14_submit_1_env_check.bat` — 环境自检

- 做什么：解析 `reproduce.yaml` → 导出环境变量到 `%TEMP%\shipreid_repro_env.bat`，再逐项检查当前预设（`final`）所需输入是否齐备。
- 输入：`.venv-sdfnet`、`SDF-Net.yml`、官方权重、赛题 `task.json`、各成员 ckpt（有 `ckpt` 字段的成员）、ship_reid_vit 的 venv 与 `best.pth`。
- 输出：控制台 OK/MISS 清单；全部必需项齐备时退出码 0。
- 失败怎么办：按 `[MISS]` 行补齐——venv 缺失回 §3；ckpt/official 缺失跑 `12_sdfnet_setup.bat`；ship_reid_vit 缺失跑 `ship_reid_vit\01_setup.bat`。
- 可用 `--preset <名>` 临时切换预设做自检。

### 4.2 `15_submit_2_sdfnet_infer.bat` — SDF-Net 成员推理

- 做什么：对当前预设中**带 `ckpt` 的成员**逐个在测试集推理（`scripts\sdfnet_inference.py`，`--batch_size 16`，cwd 切到 `SDF-Net`，因 `SDF-Net.yml` 的 `PRETRAIN_PATH` 是相对路径）。
- 输入：`logs\SDF-Net-mos\transformer_75.pth`（锚点）、`logs\SDF-Net-finetune\transformer_45.pth`、`transformer_25.pth`；`question6-data\...\task.json`。
- 输出：`sims\test\pred_mos75.json` / `pred_ep45.json` / `pred_ep25.json`（`final` 预设共 3 个成员需本地推理；`SDF_N=0` 时直接跳过）。
- 失败怎么办：检查 ckpt 路径（`ckpt` 相对 `sdf_checkpoint_dir`）、显存（OOM 降 batch）。

### 4.3 `16_submit_3_shipvit_infer.bat` — ship_reid_vit 推理

- 做什么：`final` / `anchor_plus_sv` 预设会用到 ship_reid_vit；用 `ship_reid_vit\venv` 运行 `inference.py --config config\train_vit.yaml --use_test_task --tta --rerank`。
- **默认行为：若 `ship_reid_vit\outputs\prediction.json` 已存在则直接复用、不重跑**（保证结果可比）。需要重跑时执行 `16_submit_3_shipvit_infer.bat force`。
- 输出：`ship_reid_vit\outputs\prediction.json`。
- 失败怎么办：报「venv not found」→ 跑 `ship_reid_vit\01_setup.bat`；报「checkpoint not found」→ 跑 `ship_reid_vit\03_train.bat` 或拷入 `outputs\checkpoints\best.pth`。

### 4.4 `17_submit_4_rrf_fuse.bat` — RRF 融合

- 做什么：按 `reproduce.yaml` 当前预设读取各成员预测列表，`score(c)=Σ_m w_m/(k+rank_m(c))`（`k=10`），每 query 取 top-10。
- 输入：全部成员预测（4 个文件）；query 集合一致性由脚本校验。
- 输出：`h:\Ship-Re-Identification\prediction.json`（`out_prediction: ../prediction.json`）；工作副本 `submission_rrf.json`。
- 可覆盖：`17_submit_4_rrf_fuse.bat --preset anchor_plus_sv`，或 `--weights 2 1 1 1`（个数须与预设成员数一致）。
- 失败怎么办：报成员预测缺失 → 回 15/16 号；报 query 集合不一致 → 某成员预测与 task.json 不匹配，重跑该成员推理。

### 4.5 `18_submit_5_verify.bat` — 提交格式终检

- 做什么：调用 `scripts\check_prediction.py` 逐条对照官方「直接判定无效」条件（文件名/编码、query 集合、每 query 恰好 10 个、候选去重且属 gallery、模态匹配、无额外字段、无 query 自身泄漏）。
- 输出：PASS/FAIL 报告，并回显本地验证基准（Final 0.6346 / fold0 0.6131 / fold1 0.6561）。
- 失败怎么办：**不要上传**；按 FAIL 明细回溯到 17 或成员推理。校验用显式报错而非 `assert`，`python -O` 下不会被剥离。

### 4.6 `13_sdfnet_plan.bat`（历史遗留入口）

分阶段对比编排。其 C/D 阶段依赖**已删除**的 TransOSS 侧，本就不可运行；A/B 阶段（官方权重/微调权重的 base 与 rerankqe 对比）仍可作微调入口使用。

---

## 5. 依赖关系图

```
                  12_sdfnet_setup.bat ─┐
        (建 .venv-sdfnet / 官方权重 / HOSS 数据)
                                       ▼
                    14_submit_1_env_check   [.venv-sdfnet]
                                       ▼
                    15_submit_2_sdfnet_infer [.venv-sdfnet] ──► sims\test\pred_*.json
                                       ▼
ship_reid_vit\01_setup.bat ──►  ship_reid_vit\venv
                                       ▼
                    16_submit_3_shipvit_infer [ship_reid_vit\venv] ──► ship_reid_vit\outputs\prediction.json
                                       ▼
                    17_submit_4_rrf_fuse     [.venv-sdfnet] ──► prediction.json
                                       ▼
                    18_submit_5_verify       [.venv-sdfnet]
```

- 14 / 15 / 17 / 18 号**必须用 `.venv-sdfnet`**。
- **16 号是唯一需要 `ship_reid_vit\venv` 的步骤**（若其预测已存在则跳过，不需要该环境）。
- `19_mos_retrain.bat` 依赖 `.venv-sdfnet` 与 GPU，仅在需要重训锚点时使用（配置 `SDF-Net-mos.yml`，输出 `logs\SDF-Net-mos`）。

---

## 6. 常见问题 / 注意事项

| 事项 | 说明 |
|---|---|
| 16 号默认复用 | 已有 `ship_reid_vit\outputs\prediction.json` 时不重跑；要重跑加 `force` |
| 无 Git | 项目**无 Git 仓库**，删除/改动不可逆，操作前自行备份 |
| 路径统一 | 所有相对路径一律相对工程根目录解析（`scripts\repro_cfg.py::resolve`）；bat 里不写死路径 |
| 唯一事实来源 | 改成员/权重/k 只改 `configs\reproduce.yaml` 的 `ensemble.members` / `ensemble.presets` / `ensemble.active` / `k` / `topk` |
| 成员定义 | `{name, ckpt, pred}`；`ckpt` 相对 `sdf_checkpoint_dir`（留空 = 复用已有预测、不做本地推理）；`pred` 相对工程根目录 |
| 提交文件 | 最终 = `h:\Ship-Re-Identification\prediction.json`；`submission_rrf.json` 是同内容工作副本 |
| 选模 | 用 `scripts\sdfnet_ckpt_sweep.py` 按 `Final` 排序；**严禁**按 SDF-Net 内置 val mAP / overall mAP 选模 |
| 保留门槛 | 每新增模块若 `Final` 提升 < 0.003 则删除；多成员选择必须做 fold 交叉验证（两折同时 ≥0.003 才保留） |
| 训练超参 | AdamW、`BASE_LR 1e-4`、`IMS_PER_BATCH 32`、`NUM_INSTANCE 2`、`MAX_EPOCHS 80`、每 5 epoch 存 ckpt；调度为 CosineLRScheduler（`t_initial=SOLVER.MAX_EPOCHS`，`SOLVER.STEPS` 实际未使用，延长训练只能改 `MAX_EPOCHS`） |
| 官方权重两份 | `SDF-Net.pth` 是官方 256×128 原始权重；`SDF-Net_256.pth` 是它的 256×256 衍生版（pos_embed 插值）。final 链路只用后者，两份均由 `12_sdfnet_setup.bat` 自动准备 |
| 官方模板 | `README.official.md` / `技术方案.official.md` / `路线方案.official.md` 为赛题官方起始模板（已过期），仅供参照 |

---

## 7. 参考文档

| 文档 | 内容 |
|---|---|
| `技术方案.md` | 任务与数据、评测协议与铁律、骨干/异构分支/RRF 设计、量化结果、CMAL 损失说明 |
| `路线方案.md` | 约束锚点、已完成阶段结论、负结果登记表、剩余风险、P0 优化项与下一步方向 |
| `configs\reproduce.yaml` | 路径与集成配方的唯一事实来源 |
| `RRF集成方案.md` | RRF 的推导与可靠性说明（含历史 0.6270 口径，数值口径见 `reproduce.yaml` 为准） |