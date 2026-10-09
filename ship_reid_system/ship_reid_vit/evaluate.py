"""评测入口。

模式1（整体检索评测）：python evaluate.py --ckpt outputs/checkpoints/best.pth [--ann ...]
    对给定标注集提取特征，计算整体/同模态/跨模态 mAP 与 Recall@K。

模式2（赛题提交评测）：python evaluate.py --submission --prediction prediction.json \
        --task task.json --gt ground_truth.json
    按赛题指标对提交结果评分：O2S/S2O/O2O 三个方向分别计算 R@1 与 mAP@10，
    方向得分 = 两者平均，综合得分 = 0.45*O2S + 0.45*S2O + 0.10*O2O。
    gt 格式：{"query_id": [正确 gallery image_id, ...]}。
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from config import Config
from data import ShipReIDDataset, build_transforms, ensure_dummy_data, load_task
from inference import RetrievalEngine
from utils.metrics import evaluate_retrieval, evaluate_submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ship ReID Evaluation")
    p.add_argument("--config", type=str, default="", help="自定义 yaml 配置路径")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint 路径（模式1）")
    p.add_argument("--ann", type=str, default="", help="评测标注 json（模式1，默认验证集）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10], help="Recall@K 的 K 列表（模式1）")
    # 模式2：赛题提交评测
    p.add_argument("--submission", action="store_true", help="赛题提交评测模式")
    p.add_argument("--prediction", type=str, default="", help="prediction.json 路径（模式2）")
    p.add_argument("--task", type=str, default="", help="task.json 路径（模式2）")
    p.add_argument("--gt", type=str, default="", help="ground_truth.json 路径（模式2）")
    p.add_argument(
        "--local_val", action="store_true",
        help="模式2：使用 config 中的 local_val_task_json / local_val_gt_json",
    )
    return p.parse_args()


def run_submission_eval(args) -> None:
    """按赛题指标对 prediction.json 评分（O2S/S2O/O2O 方向 + 加权综合）。"""
    if not (args.prediction and args.task and args.gt):
        raise SystemExit("模式2 需要 --prediction --task --gt 三个参数")
    with open(args.prediction, "r", encoding="utf-8") as f:
        prediction = json.load(f)
    task = load_task(args.task)
    with open(args.gt, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt_positives = {k: set(v) for k, v in gt_raw.items()}

    result = evaluate_submission(prediction, gt_positives, task["queries"], topk=10)

    print("=" * 56)
    print(f"{'方向':<6}{'Query数':<8}{'R@1':<10}{'mAP@10':<12}{'方向得分':<10}{'权重':<8}")
    for d in ("O2S", "S2O", "O2O"):
        m = result[d]
        print(
            f"{d:<6}{m['count']:<8}{m['R@1']:<10.4f}{m['mAP@10']:<12.4f}"
            f"{m['score']:<10.4f}{m['weight']:<8.2f}"
        )
    print("-" * 56)
    print(f"综合得分: {result['overall']:.4f}")
    print("=" * 56)


def main() -> None:
    args = parse_args()
    if args.submission:
        if args.local_val:
            cfg = Config.load(args.config or "config/default.yaml")
            args.task = args.task or cfg.local_val_task_json
            args.gt = args.gt or cfg.local_val_gt_json
        run_submission_eval(args)
        return

    cfg = Config.load(args.config or "config/default.yaml")
    if cfg.dummy_mode:
        ensure_dummy_data(cfg)

    ann = args.ann or cfg.val_ann
    if not ann:
        raise SystemExit(
            "模式1 需要评测标注：请传 --ann 或配置 val_ann。"
            "CSV 模式下 val_ann 为空（评测请用 evaluate.py --submission 对 prediction.json 评分）。"
        )
    ds = ShipReIDDataset(ann, transform=build_transforms(cfg, is_train=False), is_train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)

    engine = RetrievalEngine(cfg, args.ckpt, device=args.device)
    feats = engine.extract(loader)
    labels = torch.tensor(ds.labels(), dtype=torch.long)
    mods = torch.tensor(ds.modalities(), dtype=torch.long)

    metrics = evaluate_retrieval(feats, labels, mods, k_values=args.k)
    print("=" * 52)
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
