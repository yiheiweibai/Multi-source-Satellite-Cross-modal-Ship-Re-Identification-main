# Phase 1 本地验证：多尺度特征融合 + SAR 增强
# 用法：在 ship_reid_vit/venv 环境下运行
#   python scripts/phase1_localval.py --ckpt <checkpoint> --config <config> [--multiscale] [--ms_weights G C S]

import argparse
import json
import sys
from pathlib import Path

import torch

# 添加 ship_reid_vit 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "ship_reid_vit"))

from config import Config
from data import build_test_loaders, load_task
from inference import RetrievalEngine, compute_sim, indices_from_sim, save_prediction, validate_prediction
from utils.metrics import evaluate_submission


def run_localval(cfg, args):
    """在本地验证集上评估多尺度特征融合效果。"""
    print(f"Loading task: {args.task}")
    task = load_task(args.task)
    queries = task["queries"]
    gallery = task["gallery"]
    print(f"Queries: {len(queries)}, Gallery: {len(gallery)}")

    # 加载 GT
    with open(args.gt, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt_positives = {k: set(v) for k, v in gt_raw.items()}

    # 构建数据加载器
    from data import GalleryDataset, QueryDataset, build_transforms
    from torch.utils.data import DataLoader

    transform = build_transforms(cfg, is_train=False)
    query_loader = DataLoader(
        QueryDataset(queries, transform=transform),
        batch_size=getattr(cfg, "test_batch_size", 32), shuffle=False, num_workers=0,
    )
    gallery_loader = DataLoader(
        GalleryDataset(gallery, transform=transform),
        batch_size=getattr(cfg, "test_batch_size", 32), shuffle=False, num_workers=0,
    )

    # 提取特征并计算相似度
    engine = RetrievalEngine(cfg, args.ckpt, device=args.device)
    sim, q_feats, g_feats, g_mods = compute_sim(
        engine, query_loader, gallery_loader, queries, gallery,
        tta=args.tta, rerank=args.rerank,
        multiscale=args.multiscale, ms_weights=tuple(args.ms_weights),
        directional=args.directional,
    )

    # 生成预测
    indices = indices_from_sim(sim, g_mods, queries, gallery, topk=args.topk)
    prediction = {
        queries[qi]["query_id"]: [gallery[gi]["image_id"] for gi in indices[qi].tolist()]
        for qi in range(len(queries))
    }

    # 格式校验
    errors = validate_prediction(prediction, queries, gallery, topk=args.topk)
    if errors:
        print(f"格式错误: {errors[:5]}")
        return

    # 评估
    result = evaluate_submission(prediction, gt_positives, queries, topk=args.topk)

    print("\n" + "=" * 60)
    print(f"配置: multiscale={args.multiscale}, ms_weights={args.ms_weights}, "
          f"tta={args.tta}, rerank={args.rerank}, directional={args.directional}")
    print("-" * 60)
    print(f"{'方向':<6}{'Query数':<8}{'R@1':<10}{'mAP@10':<12}{'方向得分':<10}{'权重':<8}")
    for d in ("O2S", "S2O", "O2O"):
        m = result[d]
        print(f"{d:<6}{m['count']:<8}{m['R@1']:<10.4f}{m['mAP@10']:<12.4f}"
              f"{m['score']:<10.4f}{m['weight']:<8.2f}")
    print("-" * 60)
    print(f"综合得分: {result['overall']:.4f}")
    print("=" * 60)

    # 保存预测（可选）
    if args.out:
        save_prediction(prediction, args.out)
        print(f"预测已保存: {args.out}")

    return result


def main():
    p = argparse.ArgumentParser(description="Phase 1 本地验证：多尺度特征融合")
    p.add_argument("--config", type=str, required=True, help="配置文件路径")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    p.add_argument("--task", type=str, default="", help="task.json 路径（默认用 config 中的 local_val_task_json）")
    p.add_argument("--gt", type=str, default="", help="ground_truth.json 路径（默认用 config 中的 local_val_gt_json）")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--multiscale", action="store_true", help="启用多尺度特征融合")
    p.add_argument("--ms_weights", type=float, nargs=3, default=[0.5, 0.3, 0.2],
                   metavar=("G", "C", "S"), help="多尺度权重：全局 中心 条带")
    p.add_argument("--directional", action="store_true", help="方向特异性投影（Phase 4）")
    p.add_argument("--out", type=str, default="", help="输出 prediction.json 路径")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if not args.task:
        args.task = cfg.local_val_task_json
    if not args.gt:
        args.gt = cfg.local_val_gt_json

    run_localval(cfg, args)


if __name__ == "__main__":
    main()
