# -*- coding: utf-8 -*-
"""checkpoint 选优：对某个训练输出目录下所有 transformer_*.pth 逐一执行
「推理 + 赛题协议评测」，按 Final Score 排序选最优 checkpoint。

Final Score = 0.45*O2S + 0.45*S2O + 0.10*O2O，方向分 = (R@1 + mAP@10)/2
（铁律：不得按 SDF-Net 内置 val mAP / overall mAP 选模）

用法（cwd 任意）：
    python scripts/sdfnet_ckpt_sweep.py --ckpt_dir logs/SDF-Net-finetune --tag e0_256
    python scripts/sdfnet_ckpt_sweep.py --ckpt_dir logs/SDF-Net-finetune-128ep35 --tag e0_128
可选：--rerank / --qe / --tta 作用于全部 checkpoint；--max_epochs 只跑前 N 个。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SHIP_DIR = Path(__file__).resolve().parent.parent
ROOT = SHIP_DIR.parent
PY_SDF = SHIP_DIR / ".venv-sdfnet" / "Scripts" / "python.exe"
SDF_DIR = SHIP_DIR / "SDF-Net"
DATA_DIR = ROOT / "question6-data" / "traindata"
TASK_JSON = DATA_DIR / "local_val_task.json"
GT_JSON = DATA_DIR / "local_val_gt.json"
SDF_CFG = "configs/SDF-Net.yml"
OUT_ROOT = SHIP_DIR / "sims" / "sweep"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDF-Net checkpoint Final Score sweep")
    p.add_argument("--ckpt_dir", required=True, help="含 transformer_*.pth 的目录（相对工程根目录或绝对）")
    p.add_argument("--tag", required=True, help="本次扫描标签，用于输出子目录/文件名")
    p.add_argument("--task_json", default=str(TASK_JSON))
    p.add_argument("--gt_json", default=str(GT_JSON))
    p.add_argument("--config_file", default=SDF_CFG)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--qe", action="store_true")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--max_epochs", type=int, default=0, help=">0 时只评测 epoch<=该值的 checkpoint")
    p.add_argument("--only_epoch", type=int, default=0, help=">0 时只评测该 epoch 的单个 checkpoint")
    return p.parse_args()


def run(cmd: list, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")


def parse_eval(out: str) -> dict | None:
    scores = {}
    for d in ("O2S", "S2O", "O2O"):
        m = re.search(rf"^{d}\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out, re.M)
        if m:
            scores[d] = {"R@1": float(m.group(1)), "mAP@10": float(m.group(2)),
                         "score": float(m.group(3))}
    m = re.search(r"综合得分:\s*([\d.]+)", out)
    if not scores or not m:
        return None
    return {"scores": scores, "overall": float(m.group(1))}


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = SHIP_DIR / ckpt_dir
    ckpts = sorted(ckpt_dir.glob("transformer_*.pth"),
                   key=lambda p: int(re.findall(r"(\d+)", p.stem)[0]))
    if args.max_epochs > 0:
        ckpts = [c for c in ckpts if int(re.findall(r"(\d+)", c.stem)[0]) <= args.max_epochs]
    if args.only_epoch > 0:
        ckpts = [c for c in ckpts if int(re.findall(r"(\d+)", c.stem)[0]) == args.only_epoch]
    if not ckpts:
        raise SystemExit(f"未找到 checkpoint: {ckpt_dir}")

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep:{args.tag}] {len(ckpts)} 个 checkpoint -> {out_dir}", flush=True)

    records = []
    for ckpt in ckpts:
        ep = int(re.findall(r"(\d+)", ckpt.stem)[0])
        pred = out_dir / f"pred_ep{ep}.json"
        t0 = time.time()
        inf_cmd = [str(PY_SDF), str(SHIP_DIR / "scripts" / "sdfnet_inference.py"),
                   "--config_file", args.config_file, "--weight", str(ckpt),
                   "--task_json", args.task_json, "--out_prediction", str(pred),
                   "--batch_size", str(args.batch_size)]
        if args.rerank:
            inf_cmd.append("--rerank")
        if args.qe:
            inf_cmd.append("--qe")
        if args.tta:
            inf_cmd.append("--tta")
        rc, out = run(inf_cmd, SDF_DIR)
        if rc != 0:
            print(f"[sweep:{args.tag}] ep{ep} 推理失败 rc={rc}\n{out[-1500:]}", flush=True)
            records.append({"epoch": ep, "status": "INFER_FAIL"})
            continue
        inf_s = time.time() - t0

        t1 = time.time()
        eval_cmd = [str(PY_SDF), str(SHIP_DIR / "evaluate.py"), "--submission",
                    "--prediction", str(pred), "--task", args.task_json, "--gt", args.gt_json]
        rc, out = run(eval_cmd, SHIP_DIR)
        res = parse_eval(out) if rc == 0 else None
        if res is None:
            print(f"[sweep:{args.tag}] ep{ep} 评测失败 rc={rc}\n{out[-1500:]}", flush=True)
            records.append({"epoch": ep, "status": "EVAL_FAIL"})
            continue
        rec = {"epoch": ep, "status": "OK", "infer_s": round(inf_s),
               "eval_s": round(time.time() - t1), "overall": res["overall"],
               "scores": res["scores"]}
        records.append(rec)
        s = res["scores"]
        print(f"[sweep:{args.tag}] ep{ep:>3}  O2S={s['O2S']['score']:.4f} "
              f"S2O={s['S2O']['score']:.4f} O2O={s['O2O']['score']:.4f} "
              f"Final={res['overall']:.4f}  ({inf_s:.0f}s)", flush=True)

    ok = [r for r in records if r["status"] == "OK"]
    ok.sort(key=lambda r: r["overall"], reverse=True)
    result = {"tag": args.tag, "ckpt_dir": str(ckpt_dir),
              "opts": {"rerank": args.rerank, "qe": args.qe, "tta": args.tta},
              "task_json": args.task_json, "gt_json": args.gt_json,
              "best": ok[0] if ok else None, "ranking": ok, "all": records}
    with open(out_dir / "sweep_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72, flush=True)
    print(f"{'epoch':<8}{'O2S':<10}{'S2O':<10}{'O2O':<10}{'Final':<10}", flush=True)
    for r in ok:
        s = r["scores"]
        print(f"{r['epoch']:<8}{s['O2S']['score']:<10.4f}{s['S2O']['score']:<10.4f}"
              f"{s['O2O']['score']:<10.4f}{r['overall']:<10.4f}", flush=True)
    if ok:
        b = ok[0]
        print(f"最优: epoch {b['epoch']}  Final={b['overall']:.4f}", flush=True)
    print(f"结果写入: {out_dir / 'sweep_result.json'}", flush=True)


if __name__ == "__main__":
    main()