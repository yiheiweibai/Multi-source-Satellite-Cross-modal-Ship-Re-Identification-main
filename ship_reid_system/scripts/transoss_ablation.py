"""一键消融验证：SAR 预处理 / rerank+QE / 混合三组增强对比 baseline。

对同一份本地验证集 local_val_task.json 依次跑 4 个组合（复用同一划分保证横向可比）：
  1) base       : 无增强参数（baseline 参照）
  2) preprocess : --preprocess（SAR 去斑 + CLAHE）
  3) rerankqe   : --rerank --qe
  4) mixed      : --preprocess --rerank --qe
每个组合输出独立 prediction 文件（pred_ab_*.json）并调 evaluate.py --submission 评测
（解析 O2S/S2O/O2O 方向得分与综合得分），末尾打印横向对比汇总表，
供"只保留正增益项"的决策使用。

用法（在 ship_reid_system 根目录，或由 11_transoss_ab.bat 双击触发）：
    python scripts/transoss_ablation.py \
        --task_json "../赛题6-初赛/训练数据/local_val_task.json"

可选参数：
    --hoss_dir     TransOSS 仓库目录（默认 ship_reid_system/Hoss-ReID）
    --config_file  推理配置（默认 configs/hoss_transoss_competition.yml，相对 hoss_dir）
    --weight       推理权重（默认 logs/competition_transoss/transformer_200.pth，相对 hoss_dir）
    --out_dir      预测输出目录（默认 ship_reid_system 根，与 09 的 pred_local_val.json 同级）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---- 路径常量 ----
SCRIPT_DIR = Path(__file__).resolve().parent          # ship_reid_system/scripts
SHIP_DIR = SCRIPT_DIR.parent                          # ship_reid_system
REPO_ROOT = SHIP_DIR.parent                           # 仓库根

# ---- 消融组合：名字 -> transoss_inference.py 追加参数 ----
COMBOS = [
    ("base",       []),
    ("preprocess", ["--preprocess"]),
    ("rerankqe",   ["--rerank", "--qe"]),
    ("mixed",      ["--preprocess", "--rerank", "--qe"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="一键消融验证（base / preprocess / rerank+QE / mixed）")
    p.add_argument("--hoss_dir", type=str, default=str(SHIP_DIR / "Hoss-ReID"))
    p.add_argument("--config_file", type=str, default="configs/hoss_transoss_competition.yml")
    p.add_argument("--weight", type=str, default="")
    p.add_argument("--task_json", type=str, required=True, help="本地验证集 task.json")
    p.add_argument("--out_dir", type=str, default=str(SHIP_DIR))
    return p.parse_args()


def resolve_weight(hoss_dir: Path, weight: str) -> Path:
    if weight:
        w = Path(weight)
        return w if w.is_absolute() else (hoss_dir / w)
    return hoss_dir / "logs" / "competition_transoss" / "transformer_200.pth"


def run_inference(hoss_dir: Path, inf_script: Path, config_file: Path, weight: Path,
                  task_json: Path, pred_path: Path, extra_args: list[str]) -> None:
    """子进程调用 transoss_inference.py（cwd=Hoss-ReID，与 09 约定一致），生成 prediction。"""
    cmd = [
        sys.executable, str(inf_script),
        "--config_file", str(config_file),
        "--weight", str(weight),
        "--task_json", str(task_json),
        "--out_prediction", str(pred_path),
    ] + extra_args
    env = dict(os.environ)
    env["PYTHONPATH"] = str(hoss_dir) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hoss_dir), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"transoss_inference.py 推理失败: {proc.stderr[-2000:]}")
    print(proc.stdout[-1500:])


def run_eval(pred_path: Path, task_json: Path, gt_json: Path) -> tuple[dict, float]:
    """子进程调用 evaluate.py --submission，解析方向得分与综合得分（复用 fusion 脚本逻辑）。"""
    cmd = [
        sys.executable, str(SHIP_DIR / "evaluate.py"), "--submission",
        "--prediction", str(pred_path),
        "--task", str(task_json),
        "--gt", str(gt_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SHIP_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py 评测失败: {proc.stderr[-2000:]}")
    scores: dict = {}
    for d in ("O2S", "S2O", "O2O"):
        m = re.search(rf"^{d}\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", proc.stdout, re.M)
        if m:
            scores[d] = float(m.group(3))  # 方向得分（R@1 与 mAP@10 的平均）
    m = re.search(r"综合得分:\s*([\d.]+)", proc.stdout)
    overall = float(m.group(1)) if m else None
    return scores, overall


def main() -> None:
    args = parse_args()
    hoss_dir = Path(args.hoss_dir).resolve()
    config_file = (hoss_dir / args.config_file) if not Path(args.config_file).is_absolute() else Path(args.config_file).resolve()
    weight = resolve_weight(hoss_dir, args.weight)
    task_json = Path(args.task_json).resolve()
    gt_json = task_json.parent / "local_val_gt.json"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 环境检查 ----
    if not hoss_dir.exists():
        raise SystemExit(f"[ab] Hoss-ReID 目录不存在: {hoss_dir}，请先运行 08_transoss_setup.bat")
    inf_script = hoss_dir / "transoss_inference.py"
    if not inf_script.exists():
        raise SystemExit(f"[ab] 未找到 {inf_script}，请先运行 08_transoss_setup.bat")
    if not config_file.exists():
        raise SystemExit(f"[ab] 配置不存在: {config_file}")
    if not weight.exists():
        raise SystemExit(f"[ab] 权重不存在: {weight}，请先运行 07_transoss_pipeline.bat 完成训练")
    if not task_json.exists():
        raise SystemExit(f"[ab] task.json 不存在: {task_json}，请先运行 09 或 11 生成验证集")
    if not gt_json.exists():
        raise SystemExit(f"[ab] 缺少 {gt_json}，请先运行 09 或 11 生成验证集")

    print(f"[ab] 验证集: {task_json}")
    print(f"[ab] 权重:   {weight}")
    print(f"[ab] 组合:   {', '.join(c[0] for c in COMBOS)}")

    # ---- 逐组合推理 + 评测 ----
    results = []  # (name, overall, scores, pred_path)
    for name, extra in COMBOS:
        pred_path = out_dir / f"pred_ab_{name}.json"
        print(f"\n[ab] === 组合 {name}: {extra if extra else '(无增强, baseline)'} ===")
        run_inference(hoss_dir, inf_script, config_file, weight, task_json, pred_path, extra)
        scores, overall = run_eval(pred_path, task_json, gt_json)
        print(f"[ab] {name}: 综合 {overall:.4f}  O2S {scores.get('O2S', float('nan')):.4f}  "
              f"S2O {scores.get('S2O', float('nan')):.4f}  O2O {scores.get('O2O', float('nan')):.4f}")
        results.append((name, overall, scores, pred_path))

    # ---- 横向对比汇总表（含 baseline 参照）----
    print("\n" + "=" * 78)
    print(f"{'组合':<14}{'O2S':<10}{'S2O':<10}{'O2O':<10}{'综合':<10}{'相对baseline':<10}")
    print("-" * 78)
    base_overall = results[0][1]
    for name, overall, scores, _ in results:
        delta = (overall - base_overall) if (overall is not None and base_overall is not None) else None
        delta_s = f"{delta:+.4f}" if delta is not None else "n/a"
        print(f"{name:<14}{scores.get('O2S', float('nan')):<10.4f}"
              f"{scores.get('S2O', float('nan')):<10.4f}{scores.get('O2O', float('nan')):<10.4f}"
              f"{overall:<10.4f}{delta_s:<10}")
    print("=" * 78)
    print("[ab] 预测已保存: " + ", ".join(str(r[3]) for r in results))
    print("[ab] 只保留相对 baseline 为正增益的组合项，叠加进最终推理配置。")


if __name__ == "__main__":
    main()
