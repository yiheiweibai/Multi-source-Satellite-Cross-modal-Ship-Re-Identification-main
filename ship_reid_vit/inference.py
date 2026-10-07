"""推理入口（CLI 壳，复用 inference/ 包实现）。

模式1（通用检索）：python inference.py --ckpt outputs/checkpoints/best.pth
    对 JSON 标注查询/图库做 Top-K 检索，输出 outputs/retrieval_results.json。

模式2（赛题提交）：python inference.py --ckpt ... --task_json <测试包>/task.json \
        [--max_queries_per_dir N] [--max_gallery_per_mod M] [--out_prediction path]
    解析官方测试包 task.json，按 query_type 过滤候选模态（O2S->sar、S2O/O2O->optical），
    对每个 query 取相似度降序前 10 个不重复 gallery image_id，生成符合提交格式的
    prediction.json（UTF-8，顶层 JSON 对象，仅含 query_id -> [10 个 image_id]），
    并在保存前做格式自检（全覆盖 / 恰好 10 个 / 不重复 / ID 合法 / 模态匹配）。

    --max_queries_per_dir / --max_gallery_per_mod 为冒烟调试参数：
    按方向/模态均衡截取子集（0 = 全部，正式提交请勿设置）。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from config import Config
from data import ensure_dummy_data, load_task
from inference import (
    RetrievalEngine,
    retrieve_for_submission,
    save_prediction,
    topk_results,
    validate_prediction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ship ReID Inference / Retrieval")
    p.add_argument("--config", type=str, default="", help="自定义 yaml 配置路径")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    p.add_argument("--query_ann", type=str, default="", help="查询集标注 json（模式1）")
    p.add_argument("--gallery_ann", type=str, default="", help="图库标注 json（模式1）")
    p.add_argument("--task_json", type=str, default="", help="官方测试包 task.json（模式2）")
    p.add_argument(
        "--use_test_task", action="store_true",
        help="模式2：使用 config 中的 test_task_json（避免 bat 传入非 ASCII 路径）",
    )
    p.add_argument(
        "--local_val_task", action="store_true",
        help="模式2：使用 config 中的 local_val_task_json（本地验证协议）",
    )
    p.add_argument(
        "--max_queries_per_dir", type=int, default=0,
        help="模式2：每个 query_type 最多取 N 条 query（0=全部，冒烟调试用）",
    )
    p.add_argument(
        "--max_gallery_per_mod", type=int, default=0,
        help="模式2：每种模态最多取 M 个 gallery（0=全部，冒烟调试用）",
    )
    p.add_argument("--out_prediction", type=str, default="", help="模式2输出 prediction.json 路径")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    # 后处理开关
    p.add_argument("--tta", action="store_true", help="水平翻转 TTA（原图+翻转特征平均）")
    p.add_argument("--rerank", action="store_true", help="k-reciprocal 重排序")
    p.add_argument("--qe", action="store_true", help="查询扩展（top-1 gallery 特征扩展 query）")
    p.add_argument("--cluster", action="store_true", help="Gallery 聚类 + 类别中心检索")
    return p.parse_args()


def _subset_task(task: dict, max_queries_per_dir: int, max_gallery_per_mod: int) -> dict:
    """按 query_type 均衡截取 queries、按模态均衡截取 gallery（保证三方向均有样本）。"""
    if max_queries_per_dir and max_queries_per_dir > 0:
        buckets = defaultdict(list)
        for q in task["queries"]:
            buckets[q["query_type"]].append(q)
        chosen = []
        for typ in ("O2S", "S2O", "O2O"):
            chosen.extend(buckets[typ][:max_queries_per_dir])
        task["queries"] = chosen
    if max_gallery_per_mod and max_gallery_per_mod > 0:
        gb = defaultdict(list)
        for g in task["gallery"]:
            gb[g["modality"]].append(g)
        task["gallery"] = []
        for mod in ("sar", "optical"):
            task["gallery"].extend(gb[mod][:max_gallery_per_mod])
    return task


def run_prediction(cfg, args) -> None:
    task = load_task(args.task_json)
    task = _subset_task(task, args.max_queries_per_dir, args.max_gallery_per_mod)

    from data import GalleryDataset, QueryDataset, build_transforms
    from torch.utils.data import DataLoader

    transform = build_transforms(cfg, is_train=False)
    query_loader = DataLoader(
        QueryDataset(task["queries"], transform=transform),
        batch_size=getattr(cfg, "test_batch_size", 32), shuffle=False, num_workers=0,
    )
    gallery_loader = DataLoader(
        GalleryDataset(task["gallery"], transform=transform),
        batch_size=getattr(cfg, "test_batch_size", 32), shuffle=False, num_workers=0,
    )

    dist = {t: sum(1 for q in task["queries"] if q["query_type"] == t) for t in ("O2S", "S2O", "O2O")}
    print(f"[prediction] queries={len(task['queries'])} ({dist}), gallery={len(task['gallery'])}")

    engine = RetrievalEngine(cfg, args.ckpt, device=args.device)
    indices = retrieve_for_submission(
        engine, query_loader, gallery_loader, task["queries"], task["gallery"], topk=args.topk,
        tta=args.tta, rerank=args.rerank, qe=args.qe, cluster=args.cluster,
    )
    print(f"[后处理] TTA={args.tta}, rerank={args.rerank}, QE={args.qe}, cluster={args.cluster}")
    prediction = {
        task["queries"][qi]["query_id"]: [task["gallery"][gi]["image_id"] for gi in indices[qi].tolist()]
        for qi in range(len(task["queries"]))
    }

    out_path = args.out_prediction or str(Path(cfg.output_dir) / "prediction.json")
    save_prediction(prediction, out_path)
    print(f"Saved prediction.json: {out_path}")

    errors = validate_prediction(prediction, task["queries"], task["gallery"], topk=args.topk)
    print(f"[格式自检] {'通过' if not errors else '未通过'}")
    for e in errors[:20]:
        print(f"  ! {e}")
    if errors:
        print(f"  共 {len(errors)} 条问题")
    for qid in list(prediction.keys())[:2]:
        print(f"  {qid}: {prediction[qid]}")


def run_retrieval(cfg, args) -> None:
    from data import ShipReIDDataset, build_transforms
    from torch.utils.data import DataLoader

    query_ann = args.query_ann or cfg.val_ann
    gallery_ann = args.gallery_ann or cfg.val_ann
    transform = build_transforms(cfg, is_train=False)

    query_ds = ShipReIDDataset(query_ann, transform=transform, is_train=False)
    gallery_ds = ShipReIDDataset(gallery_ann, transform=transform, is_train=False)
    query_loader = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)
    gallery_loader = DataLoader(gallery_ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)
    gallery_paths = gallery_ds.paths()
    query_paths = query_ds.paths()

    engine = RetrievalEngine(cfg, args.ckpt, device=args.device)
    indices = engine.retrieve_from_dataloaders(query_loader, gallery_loader, topk=args.topk)
    results = topk_results(indices, query_paths, gallery_paths, topk=args.topk)
    from inference import save_results
    out_path = save_results(results, str(Path(cfg.output_dir) / "retrieval_results.json"))
    print(f"Saved Top-{args.topk} retrieval results: {out_path}")
    print(f"Queries: {len(query_paths)}, Gallery: {len(gallery_paths)}")


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config or "config/default.yaml")
    if not args.task_json and args.use_test_task:
        args.task_json = cfg.test_task_json
    if not args.task_json and args.local_val_task:
        args.task_json = cfg.local_val_task_json
    if cfg.dummy_mode:
        ensure_dummy_data(cfg)

    if args.task_json:
        run_prediction(cfg, args)
    else:
        run_retrieval(cfg, args)


if __name__ == "__main__":
    main()
