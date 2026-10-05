# -*- coding: utf-8 -*-
"""统一方案入口编排脚本（13_sdfnet_plan.bat 的核心逻辑）。

按阶段依次执行 A1/A2/B1/B2/C/D1~D3/D4 共 9 个组合，自动评测并汇总生成
experiments/exp_005_sdfnet_plan_compare.md 对比文档。

组合清单（全部复用固定 local_val_task.json，不重新划分；微调只执行一次，其余共享权重零训练推理）：
  A1 : SDF-Net 官方预训练权重 base 推理
  A2 : SDF-Net 官方预训练权重 + rerankqe（按 query_type 分组后处理）
  B1 : SDF-Net 微调后（默认 60 epoch，SDF_EPOCHS 可配）base 推理
  B2 : SDF-Net 微调后 + rerankqe
  C  : 现有 TransOSS 权重 + 修复后 rerankqe 重测
  D1~D3 : SDF-Net(微调后) × TransOSS 分数融合 weighted，SDF 权重 0.5/0.6/0.7
  D4 : 同上 RRF 融合

失败/跳过策略：
  - 微调权重不存在且无 CUDA        -> B1/B2/D1~D4 跳过并记录明确原因，A1/A2/C 照常执行
  - TransOSS 侧环境（.venv/Hoss-ReID/权重）缺失 -> C/D 跳过并记录原因
  - 推理/评测失败                   -> 该组合标记 FAILED，其余组合继续执行
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SHIP_DIR = Path(__file__).resolve().parent.parent          # ship_reid_system
REPO_DIR = SHIP_DIR.parent                                 # 仓库根
SCRIPT_DIR = SHIP_DIR / "scripts"
DATA_DIR = REPO_DIR / "赛题6-初赛" / "训练数据"
TASK_JSON = DATA_DIR / "local_val_task.json"
GT_JSON = DATA_DIR / "local_val_gt.json"

PY_SDF = SHIP_DIR / ".venv-sdfnet" / "Scripts" / "python.exe"
PY_TRANS = SHIP_DIR / ".venv" / "Scripts" / "python.exe"
SDF_DIR = SHIP_DIR / "SDF-Net"
TRANS_DIR = SHIP_DIR / "Hoss-ReID"
SDF_CFG = "configs/SDF-Net.yml"
SDF_OFFICIAL = SDF_DIR / "logs" / "SDF-Net" / "SDF-Net.pth"
SDF_FT_DIR = SHIP_DIR / "logs" / "SDF-Net-finetune"
SDF_FT_WEIGHT = SDF_FT_DIR / "best.pth"
TRANS_CFG = "configs/hoss_transoss.yml"
TRANS_WEIGHT = TRANS_DIR / "logs" / "competition_transoss" / "transformer_200.pth"

PLAN_DIR = SHIP_DIR / "sims" / "plan"
SIM_SDF_OFF = PLAN_DIR / "sdfnet_official"
SIM_SDF_FT = PLAN_DIR / "sdfnet_ft"
SIM_TRANS = PLAN_DIR / "transoss"
FUSED_DIR = PLAN_DIR / "fused"
RUN_LOG = PLAN_DIR / "run.log"

OUT_MD = REPO_DIR / "experiments" / "exp_005_sdfnet_plan_compare.md"

BASELINE = {"划分1": 0.4881, "划分2": 0.4831}   # TransOSS baseline（任务约定）
GAIN_TH = 0.005                                 # 正增益阈值：>0.005 才算


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDF-Net × TransOSS 统一方案编排（13_sdfnet_plan）")
    p.add_argument("--epochs", type=int, default=60, help="微调 epoch（默认 60，可配）")
    p.add_argument("--ims_per_batch", type=int, default=32, help="微调 batch size（16GB 显存建议 32，OOM 设 16）")
    p.add_argument("--ft_dir", type=str, default=str(SDF_FT_DIR), help="微调输出目录")
    p.add_argument("--task_json", type=str, default=str(TASK_JSON))
    p.add_argument("--gt_json", type=str, default=str(GT_JSON))
    p.add_argument("--sdf_official_weight", type=str, default=str(SDF_OFFICIAL))
    p.add_argument("--sdf_ft_weight", type=str, default=str(SDF_FT_WEIGHT))
    p.add_argument("--trans_weight", type=str, default=str(TRANS_WEIGHT))
    p.add_argument("--out_md", type=str, default=str(OUT_MD))
    return p.parse_args()


def log_write(logf, msg: str) -> None:
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def run_cmd(cmd: list, cwd: Path, logf, tag: str) -> tuple[bool, str]:
    logf.write(f"\n[{tag}] $ {' '.join(str(c) for c in cmd)}\n")
    logf.flush()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        logf.write(f"[{tag}] 启动失败: {e}\n")
        logf.flush()
        return False, f"启动失败: {e}"
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    tail = out[-8000:]
    logf.write(f"[{tag}] ---- 输出(尾部 {len(tail)} 字符) ----\n{tail}\n")
    logf.write(f"[{tag}] rc={proc.returncode} elapsed={time.time() - t0:.0f}s\n")
    logf.flush()
    if proc.returncode != 0:
        return False, out
    return True, out


def parse_eval(out: str) -> dict | None:
    scores = {}
    for d in ("O2S", "S2O", "O2O"):
        m = re.search(rf"^{d}\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out, re.M)
        if m:
            scores[d] = {"R@1": float(m.group(1)), "mAP": float(m.group(2)), "score": float(m.group(3))}
    m = re.search(r"综合得分:\s*([\d.]+)", out)
    overall = float(m.group(1)) if m else None
    if not scores or overall is None:
        return None
    return {"scores": scores, "overall": overall}


def run_eval(pred_path: Path, logf) -> dict | None:
    cmd = [str(PY_SDF), str(SHIP_DIR / "evaluate.py"), "--submission",
           "--prediction", str(pred_path), "--task", str(TASK_JSON), "--gt", str(GT_JSON)]
    ok, out = run_cmd(cmd, SHIP_DIR, logf, f"eval:{pred_path.name}")
    if not ok:
        return None
    return parse_eval(out)


def sdf_infer(weight: Path, out_pred: Path, logf, tag: str, save_sim: str = "",
              rerank: bool = False, qe: bool = False) -> tuple[bool, str]:
    cmd = [str(PY_SDF), str(SCRIPT_DIR / "sdfnet_inference.py"),
           "--config_file", SDF_CFG, "--weight", str(weight),
           "--task_json", str(TASK_JSON), "--out_prediction", str(out_pred)]
    if save_sim:
        cmd += ["--save_sim", save_sim]
    if rerank:
        cmd.append("--rerank")
    if qe:
        cmd.append("--qe")
    return run_cmd(cmd, SDF_DIR, logf, tag)


def trans_infer(weight: Path, out_pred: Path, logf, tag: str, save_sim: str = "",
                rerank: bool = False, qe: bool = False) -> tuple[bool, str]:
    cmd = [str(PY_TRANS), "transoss_inference.py",
           "--config_file", TRANS_CFG, "--weight", str(weight),
           "--task_json", str(TASK_JSON), "--out_prediction", str(out_pred)]
    if save_sim:
        cmd += ["--save_sim", save_sim]
    if rerank:
        cmd.append("--rerank")
    if qe:
        cmd.append("--qe")
    return run_cmd(cmd, TRANS_DIR, logf, tag)


def run_finetune(epochs: int, ims: int, ft_dir: Path, logf) -> tuple[bool, str]:
    cmd = [str(PY_SDF), "train.py", "--config_file", SDF_CFG,
           "MODEL.PRETRAIN_CHOICE", "imagenet",
           "MODEL.PRETRAIN_PATH", "logs/SDF-Net/SDF-Net.pth",
           "DATASETS.ROOT_DIR", "../data",
           "SOLVER.MAX_EPOCHS", str(epochs),
           "SOLVER.IMS_PER_BATCH", str(ims),
           "OUTPUT_DIR", str(ft_dir)]
    return run_cmd(cmd, SDF_DIR, logf, "finetune")


def cuda_available(logf) -> bool:
    cmd = [str(PY_SDF), "-c", "import torch; print(torch.cuda.is_available())"]
    ok, out = run_cmd(cmd, SHIP_DIR, logf, "cuda-check")
    if not ok:
        return False
    return out.strip().endswith("True")


def run_fusion(logf) -> dict:
    """D1~D4：SDF-Net(微调后 base sim) × TransOSS(base sim) 分数融合。"""
    sys.path.insert(0, str(SCRIPT_DIR))
    import fuse_sdfnet_transoss as fuse_mod  # noqa: E402

    sim_a, meta_a = fuse_mod.load_pair(str(SIM_SDF_FT / "sim.pt"), str(SIM_SDF_FT / "meta.json"))
    sim_b, meta_b = fuse_mod.load_pair(str(SIM_TRANS / "sim.pt"), str(SIM_TRANS / "meta.json"))
    ref_q, ref_g = meta_a["queries"], meta_a["gallery"]
    sim_a2 = fuse_mod.align(sim_a, meta_a, ref_q, ref_g)
    sim_b2 = fuse_mod.align(sim_b, meta_b, ref_q, ref_g)
    log_write(logf, f"[fusion] aligned sim_a {tuple(sim_a2.shape)}, sim_b {tuple(sim_b2.shape)}")

    wa = fuse_mod.minmax_rows(sim_a2, ref_q, ref_g)
    wb = fuse_mod.minmax_rows(sim_b2, ref_q, ref_g)
    ra = fuse_mod.rrf_rows(sim_a2, ref_q, ref_g)
    rb = fuse_mod.rrf_rows(sim_b2, ref_q, ref_g)

    results: dict = {}
    w_map = {1: 0.5, 2: 0.6, 3: 0.7}
    for idx, w in w_map.items():
        fused = w * wa + (1 - w) * wb
        pred = fuse_mod.build_prediction(fused, ref_q, ref_g)
        p = FUSED_DIR / f"pred_D{idx}_weighted_w{w:.1f}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        results[f"D{idx}"] = run_eval(p, logf)
        log_write(logf, f"[fusion D{idx} weighted w_sdf={w}] overall="
                  f"{results[f'D{idx}']['overall'] if results[f'D{idx}'] else 'FAIL'}")

    fused = ra + rb
    pred = fuse_mod.build_prediction(fused, ref_q, ref_g)
    p = FUSED_DIR / "pred_D4_rrf.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)
    results["D4"] = run_eval(p, logf)
    log_write(logf, f"[fusion D4 rrf] overall={results['D4']['overall'] if results['D4'] else 'FAIL'}")
    return results


def fmt_num(v) -> str:
    return f"{v:.4f}" if v is not None else "-"


def build_markdown(stages: list, fusion: dict | None, args: argparse.Namespace,
                   skipped: list) -> str:
    lines = []
    lines.append("# exp_005：SDF-Net × TransOSS 统一方案对比（13_sdfnet_plan.bat）\n")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 验证集：`local_val_task.json`（固定复用，不重新划分）；评测协议 `evaluate.py --submission`（O2S/S2O/O2O 方向分 = R@1 与 mAP@10 平均，综合 = 0.45*O2S + 0.45*S2O + 0.10*O2O）")
    lines.append(f"- TransOSS baseline：划分1 = **0.4881**、划分2 = **0.4831**（exp_001 登记）；Δ 以保守基线 0.4831 为主参考，同时列出相对 0.4881 的差值")
    lines.append(f"- 正增益判定：综合得分相对保守基线 Δ **> 0.005** 才标记为「正增益」（对齐 README 8.12 决策原则）")
    lines.append("")
    lines.append("## 一、组合清单与执行状态\n")
    lines.append("| 组合 | 配置说明 | 状态 | 说明 |")
    lines.append("|---|---|---|---|")
    for key, desc, status, detail in stages:
        lines.append(f"| {key} | {desc} | {status} | {detail} |")
    if skipped:
        lines.append("")
        lines.append("**跳过项说明**：")
        for s in skipped:
            lines.append(f"- {s}")
    lines.append("")
    lines.append("## 二、完整结果表\n")
    lines.append("| 组合 | 配置 | O2S R@1 | O2S mAP | S2O R@1 | S2O mAP | O2O R@1 | O2O mAP | 综合得分 | Δ vs 0.4831 | Δ vs 0.4881 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, desc, status, detail in stages:
        if status not in ("OK",):
            continue
        ev = detail.get("eval") if isinstance(detail, dict) else None
        if not ev:
            continue
        sc = ev["scores"]
        o = ev["overall"]
        d2 = o - BASELINE["划分2"]
        d1 = o - BASELINE["划分1"]
        mark = "-"
        if d2 > GAIN_TH:
            mark = "**正增益**"
            d2s = f"**{d2:+.4f}**"
        elif d2 > 0:
            mark = "微弱"
            d2s = f"{d2:+.4f}"
        else:
            d2s = f"{d2:+.4f}"
        lines.append(
            f"| {key} | {desc} | {fmt_num(sc['O2S']['R@1'])} | {fmt_num(sc['O2S']['mAP'])} | "
            f"{fmt_num(sc['S2O']['R@1'])} | {fmt_num(sc['S2O']['mAP'])} | "
            f"{fmt_num(sc['O2O']['R@1'])} | {fmt_num(sc['O2O']['mAP'])} | "
            f"{fmt_num(o)} | {d2s} | {d1:+.4f} | {mark} |"
        )
    lines.append("")
    lines.append("## 三、增益分析与推荐组合\n")
    rows = []
    for key, desc, status, detail in stages:
        if status != "OK" or not isinstance(detail, dict) or not detail.get("eval"):
            continue
        rows.append((key, desc, detail["eval"]["overall"]))
    rows.sort(key=lambda r: r[2], reverse=True)
    if rows:
        lines.append("综合得分从高到低：")
        for i, (key, desc, o) in enumerate(rows, 1):
            d2 = o - BASELINE["划分2"]
            tag = "正增益" if d2 > GAIN_TH else ("微弱" if d2 > 0 else "无增益")
            lines.append(f"{i}. **{key}**（{desc}）：综合 **{o:.4f}**，Δ={d2:+.4f} → {tag}")
        best_key, best_desc, best_o = rows[0]
        d2 = best_o - BASELINE["划分2"]
        if d2 > GAIN_TH:
            lines.append("")
            lines.append(f"**推荐提交组合：{best_key}**（{best_desc}，综合 {best_o:.4f}，"
                         f"相对保守基线 Δ={d2:+.4f}，正增益 > 0.005）。")
        else:
            lines.append("")
            lines.append(f"**当前无正增益组合（Δ ≤ 0.005）**：最高为 {best_key}（综合 {best_o:.4f}，"
                         f"Δ={d2:+.4f}）。建议保留 TransOSS baseline 配置，不启用收益不足以覆盖风险的改动。")
        lines.append("")
        lines.append("**决策建议**：")
        lines.append("- 仅保留综合得分相对保守基线 **Δ > 0.005** 的组合进入最终提交配置；0 < Δ ≤ 0.005 视为噪声，不启用。")
        lines.append("- 若 rerankqe（A2/B2/C）与 base（A1/B1）相比无明显增益，最终推理可不启用 rerank/QE 以降低耗时。")
        lines.append("- 若 D1~D4 融合增益明显，注意最终提交需同时具备 SDF-Net 与 TransOSS 两套推理产物，复现成本更高，收益不明显时优先单模型方案。")
        lines.append("- Public 是最终裁判：本地验证集的 Δ 仅用于筛选候选，最终以测试集提交分数为准。")
    else:
        lines.append("无可用成绩（所有组合失败或跳过），请查看 run.log 定位问题。")
    lines.append("")
    lines.append("## 四、复现命令\n")
    lines.append("```bat")
    lines.append("cd /d D:\\Projects\\Multi-source Satellite Cross-modal Ship Re-Identification\\ship_reid_system")
    lines.append("13_sdfnet_plan.bat                    :: 默认：微调 60 epoch")
    lines.append("set SDF_EPOCHS=100 && 13_sdfnet_plan.bat  :: 自定义微调 epoch")
    lines.append("```")
    lines.append("")
    lines.append(f"- 中间产物（prediction/sim/run.log）：`ship_reid_system/sims/plan/`")
    lines.append(f"- 本对比文档：`experiments/exp_005_sdfnet_plan_compare.md`")
    lines.append("- 一次性环境准备仍为 `12_sdfnet_setup.bat`（SDF-Net 侧）与 `08_transoss_setup.bat`（TransOSS 侧）")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    SIM_SDF_OFF.mkdir(parents=True, exist_ok=True)
    SIM_SDF_FT.mkdir(parents=True, exist_ok=True)
    SIM_TRANS.mkdir(parents=True, exist_ok=True)

    with open(RUN_LOG, "w", encoding="utf-8") as logf:
        log_write(logf, f"===== SDF-Net × TransOSS 统一方案编排开始 {datetime.now()} =====")
        log_write(logf, f"task_json={TASK_JSON}\ngt_json={GT_JSON}")

        sdf_ok = PY_SDF.exists() and SDF_DIR.exists() and SDF_OFFICIAL.exists() \
            and (SDF_DIR / "configs" / "SDF-Net.yml").exists() and TASK_JSON.exists() and GT_JSON.exists()
        if not sdf_ok:
            log_write(logf, "[env] SDF-Net 侧环境不完整：需要 .venv-sdfnet / SDF-Net 仓库 / 官方权重 / local_val_task.json。请先运行 12_sdfnet_setup.bat。")
            log_write(logf, "===== 编排中止 =====")
            return 1

        trans_ok = PY_TRANS.exists() and TRANS_DIR.exists() and TRANS_WEIGHT.exists()
        if not trans_ok:
            log_write(logf, "[env] TransOSS 侧环境不完整（.venv / Hoss-ReID / transformer_200.pth），C/D 组合将跳过。")

        stages = []
        skipped = []
        fusion: dict | None = None

        # ---- A1 / A2：官方权重零训练 ----
        p_a1 = PLAN_DIR / "pred_A1_sdf_official_base.json"
        ok, _ = sdf_infer(SDF_OFFICIAL, p_a1, logf, "A1")
        ev_a1 = run_eval(p_a1, logf) if ok else None
        stages.append(("A1", "SDF-Net 官方权重 base", "OK" if ev_a1 else "FAILED",
                       {"eval": ev_a1} if ev_a1 else "推理或评测失败，见 run.log"))

        p_a2 = PLAN_DIR / "pred_A2_sdf_official_rerankqe.json"
        ok, _ = sdf_infer(SDF_OFFICIAL, p_a2, logf, "A2", rerank=True, qe=True)
        ev_a2 = run_eval(p_a2, logf) if ok else None
        stages.append(("A2", "SDF-Net 官方权重 + rerankqe", "OK" if ev_a2 else "FAILED",
                       {"eval": ev_a2} if ev_a2 else "推理或评测失败，见 run.log"))

        # ---- 微调（仅一次）----
        ft_weight = Path(args.sdf_ft_weight)
        if ft_weight.exists():
            log_write(logf, f"[finetune] 微调权重已存在，直接复用：{ft_weight}")
        elif cuda_available(logf):
            log_write(logf, f"[finetune] 开始微调（EPOCHS={args.epochs}, IMS={args.ims_per_batch}），仅此一次，后续组合共享权重。")
            ok, _ = run_finetune(args.epochs, args.ims_per_batch, Path(args.ft_dir), logf)
            if ok and ft_weight.exists():
                log_write(logf, f"[finetune] 微调完成：{ft_weight}")
            else:
                log_write(logf, "[finetune] 微调失败，B1/B2/D1~D4 将跳过。")
        else:
            log_write(logf, "[finetune] 无 CUDA 且微调权重不存在，B1/B2/D1~D4 跳过（推理可在 CPU 上继续 A/C）。")

        ft_ready = ft_weight.exists()
        if not ft_ready:
            skipped.append("B1/B2/D1~D4：微调权重不存在且无法训练（无 CUDA 或微调失败），跳过。")

        # ---- B1 / B2：微调后推理 ----
        if ft_ready:
            p_b1 = PLAN_DIR / "pred_B1_sdf_ft_base.json"
            ok, _ = sdf_infer(ft_weight, p_b1, logf, "B1", save_sim=str(SIM_SDF_FT))
            ev_b1 = run_eval(p_b1, logf) if ok else None
            stages.append(("B1", "SDF-Net 微调后 base", "OK" if ev_b1 else "FAILED",
                           {"eval": ev_b1} if ev_b1 else "推理或评测失败，见 run.log"))

            p_b2 = PLAN_DIR / "pred_B2_sdf_ft_rerankqe.json"
            ok, _ = sdf_infer(ft_weight, p_b2, logf, "B2", rerank=True, qe=True)
            ev_b2 = run_eval(p_b2, logf) if ok else None
            stages.append(("B2", "SDF-Net 微调后 + rerankqe", "OK" if ev_b2 else "FAILED",
                           {"eval": ev_b2} if ev_b2 else "推理或评测失败，见 run.log"))
        else:
            stages.append(("B1", "SDF-Net 微调后 base", "SKIP", "微调权重不可用"))
            stages.append(("B2", "SDF-Net 微调后 + rerankqe", "SKIP", "微调权重不可用"))

        # ---- C / C0：TransOSS 重测与 base sim ----
        if trans_ok:
            # C0：base sim（D 融合前置，不进结果表）
            p_t0 = PLAN_DIR / "pred_C0_trans_base_sim.json"
            ok, _ = trans_infer(TRANS_WEIGHT, p_t0, logf, "C0", save_sim=str(SIM_TRANS))
            if not ok:
                skipped.append("D1~D4：TransOSS base sim 导出失败，跳过。")
            # C：rerankqe 重测
            p_c = PLAN_DIR / "pred_C_trans_rerankqe.json"
            ok, _ = trans_infer(TRANS_WEIGHT, p_c, logf, "C", rerank=True, qe=True)
            ev_c = run_eval(p_c, logf) if ok else None
            stages.append(("C", "TransOSS 现有权重 + rerankqe 重测", "OK" if ev_c else "FAILED",
                           {"eval": ev_c} if ev_c else "推理或评测失败，见 run.log"))
        else:
            stages.append(("C", "TransOSS 现有权重 + rerankqe 重测", "SKIP", "TransOSS 侧环境不完整"))
            skipped.append("C/D1~D4：TransOSS 侧环境不完整（.venv / Hoss-ReID / transformer_200.pth），跳过。")

        # ---- D1~D4：融合 ----
        if ft_ready and trans_ok and (SIM_SDF_FT / "sim.pt").exists() and (SIM_TRANS / "sim.pt").exists():
            try:
                fusion = run_fusion(logf)
            except Exception as e:  # noqa: BLE001
                log_write(logf, f"[fusion] 融合执行异常：{e}")
                fusion = None
            for idx, desc in ((1, "SDF(微调后) × TransOSS weighted 0.5"),
                              (2, "SDF(微调后) × TransOSS weighted 0.6"),
                              (3, "SDF(微调后) × TransOSS weighted 0.7"),
                              (4, "SDF(微调后) × TransOSS RRF")):
                ev = (fusion or {}).get(f"D{idx}")
                if ev:
                    stages.append((f"D{idx}", desc, "OK", {"eval": ev}))
                else:
                    stages.append((f"D{idx}", desc, "FAILED", "融合评测失败，见 run.log"))
        else:
            for idx, desc in ((1, "SDF(微调后) × TransOSS weighted 0.5"),
                              (2, "SDF(微调后) × TransOSS weighted 0.6"),
                              (3, "SDF(微调后) × TransOSS weighted 0.7"),
                              (4, "SDF(微调后) × TransOSS RRF")):
                stages.append((f"D{idx}", desc, "SKIP", "前置缺失（微调权重或 TransOSS sim）"))

        # ---- 生成对比文档 ----
        md = build_markdown(stages, fusion, args, skipped)
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        log_write(logf, f"[done] 对比文档已生成：{OUT_MD}")
        log_write(logf, f"===== 编排结束 {datetime.now()} =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
