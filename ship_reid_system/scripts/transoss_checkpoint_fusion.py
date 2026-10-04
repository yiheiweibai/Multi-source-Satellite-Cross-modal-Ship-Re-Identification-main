"""多 checkpoint 批量本地验证 + RRF 排名融合。

对 Hoss-ReID 微调产生的 transformer_*.pth 逐个在本地验证集推理并评测，
再按 RRF（Reciprocal Rank Fusion）融合 top_k 个最优 checkpoint 的排名，
输出对比汇总表与融合结果 pred_fusion.json。

用法（在 ship_reid_system 根目录，或由 10_transoss_fusion.bat 双击触发）：
    python scripts/transoss_checkpoint_fusion.py \
        --task_json "../赛题6-初赛/训练数据/local_val_task.json"

可选参数：
    --preprocess   提取特征时对 SAR 图像启用预处理（去斑 + CLAHE）
    --top_k 3      融合时取综合得分最高的 N 个 checkpoint
    --hoss_dir     TransOSS 仓库目录（默认 ship_reid_system/Hoss-ReID）
    --weight_dir   checkpoint 目录（默认 Hoss-ReID/logs/competition_transoss）
    --out_dir      输出目录（默认仓库根目录）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

# ---- 路径常量 ----
SCRIPT_DIR = Path(__file__).resolve().parent          # ship_reid_system/scripts
SHIP_DIR = SCRIPT_DIR.parent                          # ship_reid_system
REPO_ROOT = SHIP_DIR.parent                           # 仓库根


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多 checkpoint 批量验证 + RRF 融合")
    p.add_argument("--hoss_dir", type=str, default=str(SHIP_DIR / "Hoss-ReID"))
    p.add_argument("--config_file", type=str, default="configs/hoss_transoss_competition.yml")
    p.add_argument("--weight_dir", type=str, default="")
    p.add_argument("--task_json", type=str, required=True, help="本地验证集 task.json")
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT))
    p.add_argument("--top_k", type=int, default=3, help="RRF 融合使用的最优 checkpoint 数")
    p.add_argument("--preprocess", action="store_true", help="SAR 图像预处理（去斑+CLAHE）")
    return p.parse_args()


def resolve_weight_dir(hoss_dir: Path, weight_dir: str) -> Path:
    if weight_dir:
        wd = Path(weight_dir)
        return wd if wd.is_absolute() else (hoss_dir / wd)
    return hoss_dir / "logs" / "competition_transoss"


def epoch_of(path: Path) -> int:
    m = re.search(r"transformer_(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else 0


def run_eval(pred_path: Path, task_json: Path, gt_json: Path) -> tuple[dict, float]:
    """子进程调用 evaluate.py --submission，解析方向得分与综合得分。"""
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
            scores[d] = float(m.group(3))
    m = re.search(r"综合得分:\s*([\d.]+)", proc.stdout)
    overall = float(m.group(1)) if m else None
    return scores, overall


def main() -> None:
    args = parse_args()
    hoss_dir = Path(args.hoss_dir).resolve()
    weight_dir = resolve_weight_dir(hoss_dir, args.weight_dir)
    task_json = Path(args.task_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_json = task_json.parent / "local_val_gt.json"

    if not hoss_dir.exists():
        raise SystemExit(f"[fusion] Hoss-ReID 目录不存在: {hoss_dir}，请先运行 08_transoss_setup.bat")
    if not weight_dir.exists():
        raise SystemExit(f"[fusion] 权重目录不存在: {weight_dir}，请先运行 07_transoss_pipeline.bat 完成训练")
    if not task_json.exists():
        raise SystemExit(f"[fusion] task.json 不存在: {task_json}，请先运行 09_local_val_transoss.bat 生成验证集")
    if not gt_json.exists():
        raise SystemExit(f"[fusion] 缺少 {gt_json}，请先运行 09_local_val_transoss.bat 生成验证集")

    # ---- 注入 TransOSS 依赖并导入模块 ----
    sys.path.insert(0, str(hoss_dir))
    sys.path.insert(0, str(SCRIPT_DIR))
    from config import cfg                                          # noqa: E402
    from model import make_model                                    # noqa: E402
    from datasets.make_dataloader import make_dataloader            # noqa: E402
    import transoss_inference                                       # noqa: E402

    cfg.merge_from_file(str(hoss_dir / args.config_file))
    cfg.freeze()

    # 加载任务与模态映射（与 transoss_inference 保持一致）
    task = transoss_inference.load_task(str(task_json))
    queries, gallery = task["queries"], task["gallery"]
    print(f"queries: {len(queries)}, gallery: {len(gallery)}")
    MOD_ID = {"optical": 0, "sar": 1}
    QTYPE_TO_QUERY_MOD = {"O2S": "optical", "S2O": "sar", "O2O": "optical"}
    QTYPE_TO_TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}

    q_paths = [q["image_path_abs"] for q in queries]
    q_mods = [MOD_ID[QTYPE_TO_QUERY_MOD[q["query_type"]]] for q in queries]
    g_paths = [g["image_path_abs"] for g in gallery]
    g_mods = [MOD_ID[g["modality"]] for g in gallery]
    g_abs = [str(Path(g["image_path_abs"]).resolve()) for g in gallery]

    # 每个 query 的合法候选（目标模态 + 排除自身），供单模型 top10 与 RRF 复用
    valid_cands: list[list[int]] = []
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        q_abs = str(Path(q["image_path_abs"]).resolve())
        cand = [i for i in cand if g_abs[i] != q_abs]
        if len(cand) < 10:
            raise RuntimeError(f"query {q['query_id']} 候选仅 {len(cand)} 个，不足 10")
        valid_cands.append(cand)

    # 模型骨架（分类头被忽略，只取特征提取部分）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _, num_query, num_classes, camera_num = make_dataloader(cfg)
    _ = (train_loader, num_query)

    # ---- 逐个 checkpoint 提取特征 + 推理 + 评测 ----
    weights = sorted(weight_dir.glob("transformer_*.pth"), key=epoch_of)
    if not weights:
        raise SystemExit(f"[fusion] 未找到 transformer_*.pth: {weight_dir}")
    print(f"发现 {len(weights)} 个 checkpoint: {[w.name for w in weights]}")

    results = []  # (epoch, overall, scores, pred_path, rank_map)
    for w in weights:
        epoch = epoch_of(w)
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
        model.load_param(str(w))
        model = model.to(device)
        model.eval()
        print(f"[fusion] 提取特征: {w.name} (preprocess={args.preprocess})")
        q_feats = transoss_inference.extract_features(
            model, q_paths, q_mods, cfg, device, tta=False, preprocess=args.preprocess)
        g_feats = transoss_inference.extract_features(
            model, g_paths, g_mods, cfg, device, tta=False, preprocess=args.preprocess)
        sim = torch.matmul(q_feats, g_feats.t())

        pred: dict = {}
        rank_map: dict = {}
        for qi, q in enumerate(queries):
            cand = valid_cands[qi]
            top_idx = sim[qi, torch.tensor(cand)].topk(10).indices.tolist()
            top_gids = [gallery[cand[t]]["image_id"] for t in top_idx]
            pred[q["query_id"]] = top_gids
            rank_map[q["query_id"]] = {gid: t + 1 for t, gid in enumerate(top_gids)}

        pred_path = out_dir / f"pred_ckpt_{epoch}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)

        scores, overall = run_eval(pred_path, task_json, gt_json)
        print(f"  {w.name}: 综合 {overall:.4f}  O2S {scores.get('O2S', float('nan')):.4f}  "
              f"S2O {scores.get('S2O', float('nan')):.4f}  O2O {scores.get('O2O', float('nan')):.4f}")
        results.append((epoch, overall, scores, pred_path, rank_map))

    # ---- 选优 + RRF 融合 ----
    results.sort(key=lambda r: r[1], reverse=True)  # 按综合得分降序
    top_models = results[: args.top_k]
    print(f"[fusion] 选优 top {len(top_models)}: {[r[0] for r in top_models]}")

    fusion_scores: dict = defaultdict(lambda: defaultdict(float))
    for epoch, _, _, _, rank_map in top_models:
        for qid, rmap in rank_map.items():
            for gid, rank in rmap.items():
                fusion_scores[qid][gid] += 1.0 / (60.0 + rank)

    pred_fusion: dict = {}
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        cand_ids = [gallery[i]["image_id"] for i in valid_cands[qi]]
        scored = [(fusion_scores[qid].get(gid, 0.0), gid) for gid in cand_ids]
        scored.sort(key=lambda x: -x[0])
        pred_fusion[qid] = [gid for _, gid in scored[:10]]

    fusion_path = out_dir / "pred_fusion.json"
    with open(fusion_path, "w", encoding="utf-8") as f:
        json.dump(pred_fusion, f, ensure_ascii=False, indent=2)
    fus_scores, fus_overall = run_eval(fusion_path, task_json, gt_json)

    # ---- 对比汇总表 ----
    print("=" * 78)
    print(f"{'模型':<24}{'O2S':<10}{'S2O':<10}{'O2O':<10}{'综合':<10}")
    print("-" * 78)
    for epoch, overall, scores, _, _ in results:
        name = f"ckpt_{epoch}"
        print(f"{name:<24}{scores.get('O2S', float('nan')):<10.4f}"
              f"{scores.get('S2O', float('nan')):<10.4f}{scores.get('O2O', float('nan')):<10.4f}"
              f"{overall:<10.4f}")
    print("-" * 78)
    print(f"{'RRF融合(top' + str(len(top_models)) + ')':<24}"
          f"{fus_scores.get('O2S', float('nan')):<10.4f}"
          f"{fus_scores.get('S2O', float('nan')):<10.4f}"
          f"{fus_scores.get('O2O', float('nan')):<10.4f}"
          f"{fus_overall:<10.4f}")
    print("=" * 78)
    print(f"[fusion] 单模型预测已保存: {[str(r[3]) for r in results]}")
    print(f"[fusion] 融合预测已保存: {fusion_path}")


if __name__ == "__main__":
    main()
