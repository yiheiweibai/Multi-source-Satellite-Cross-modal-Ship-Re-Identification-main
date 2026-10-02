"""赛题提交结果生成：按 task.json 生成 prediction.json 并进行格式自检。

提交要求（赛题）：
- UTF-8，顶层 JSON 对象；每条 query_id 恰好返回 10 个互不重复的合法 gallery image_id；
- 候选模态必须匹配 query_type：O2S -> sar、S2O -> optical、O2O -> optical；
- 数组顺序为相似度降序（第 1 位最相似）；不得含相似度/路径/模态等额外字段。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MODALITY_ID, build_test_loaders, load_task
from data.test_dataset import QUERY_TYPE_TO_MODALITY
from inference.postprocess import k_reciprocal_rerank, query_expansion
from inference.retrieval import RetrievalEngine


@torch.no_grad()
def retrieve_for_submission(
    engine: RetrievalEngine,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    queries: List[Dict],
    gallery: List[Dict],
    topk: int = 10,
    tta: bool = False,
    rerank: bool = False,
    qe: bool = False,
) -> torch.Tensor:
    """检索并强制候选模态过滤，返回 (Q, topk) 的 gallery 全局索引（相似度降序）。

    Args:
        tta: 水平翻转 TTA（原图+翻转特征平均）
        rerank: k-reciprocal 重排序
        qe: 查询扩展（用 top-1 gallery 特征扩展 query）
    """
    q_feats = engine.extract(query_loader, tta=tta)   # (Q, D)
    g_feats = engine.extract(gallery_loader, tta=tta)  # (G, D)
    g_mods = torch.as_tensor([MODALITY_ID[g["modality"]] for g in gallery], dtype=torch.long)

    # 查询扩展（在重排序之前）
    if qe:
        q_feats = query_expansion(q_feats, g_feats, topk=1, alpha=0.5)

    # 计算相似度
    if rerank:
        sim = k_reciprocal_rerank(q_feats, g_feats, k1=20, k2=6, lambda_value=0.3)
    else:
        sim = torch.matmul(q_feats, g_feats.t())    # (Q, G)

    indices = torch.full((q_feats.size(0), topk), -1, dtype=torch.long)
    for qi, query in enumerate(queries):
        target_mod = QUERY_TYPE_TO_MODALITY[query["query_type"]]
        mask = g_mods == MODALITY_ID[target_mod]
        cand = mask.nonzero().squeeze(1)
        if cand.numel() < topk:
            raise RuntimeError(
                f"query {query['query_id']} 目标模态候选仅 {cand.numel()} 个，不足 {topk}"
            )
        top = sim[qi, cand].topk(topk).indices
        indices[qi] = cand[top]
    return indices


def build_prediction(
    cfg,
    task_json_path: str,
    ckpt_path: str,
    device: str = "cpu",
    max_queries: int = 0,
    max_gallery: int = 0,
    topk: int = 10,
    num_workers: int = 0,
    tta: bool = False,
    rerank: bool = False,
    qe: bool = False,
) -> Tuple[Dict[str, List[str]], List[Dict], List[Dict]]:
    """生成提交字典 {query_id: [10 个 gallery image_id]}。

    返回 (prediction, queries_subset, gallery_subset)，
    queries_subset / gallery_subset 供 validate_prediction 做自检。
    """
    query_loader, gallery_loader, queries, gallery = build_test_loaders(
        cfg, task_json_path, max_queries=max_queries, max_gallery=max_gallery
    )
    engine = RetrievalEngine(cfg, ckpt_path, device=device)
    indices = retrieve_for_submission(
        engine, query_loader, gallery_loader, queries, gallery, topk,
        tta=tta, rerank=rerank, qe=qe,
    )

    prediction: Dict[str, List[str]] = {}
    for qi, query in enumerate(queries):
        prediction[query["query_id"]] = [gallery[gi]["image_id"] for gi in indices[qi].tolist()]
    return prediction, queries, gallery


def validate_prediction(
    prediction: Dict[str, List[str]],
    queries: List[Dict],
    gallery: List[Dict],
    topk: int = 10,
) -> List[str]:
    """提交格式自检，返回错误列表（空 = 通过）。"""
    errors: List[str] = []
    valid_ids = {g["image_id"] for g in gallery}
    id2mod = {g["image_id"]: g["modality"] for g in gallery}
    qid2type = {q["query_id"]: q["query_type"] for q in queries}

    # 1) 覆盖全部 query / 无额外 query
    missing = [q["query_id"] for q in queries if q["query_id"] not in prediction]
    if missing:
        errors.append(f"缺少 query：{missing[:10]} 等 {len(missing)} 条")
    extra = [k for k in prediction if k not in qid2type]
    if extra:
        errors.append(f"存在额外 query：{extra[:10]} 等 {len(extra)} 条")

    for qid in qid2type:
        cands = prediction.get(qid, [])
        # 2) 恰好 topk 个
        if len(cands) != topk:
            errors.append(f"{qid}: 候选数量 {len(cands)} != {topk}")
        # 3) 不重复
        if len(set(cands)) != len(cands):
            errors.append(f"{qid}: 候选重复")
        # 4) ID 合法
        invalid = [c for c in cands if c not in valid_ids]
        if invalid:
            errors.append(f"{qid}: 非法 gallery ID {invalid[:5]}")
        # 5) 模态匹配
        need = QUERY_TYPE_TO_MODALITY[qid2type[qid]]
        wrong = [c for c in cands if id2mod.get(c) != need]
        if wrong:
            errors.append(f"{qid}: 模态错误候选 {wrong[:5]}（应为 {need}）")
    return errors


def save_prediction(prediction: Dict[str, List[str]], out_path: str) -> str:
    """写 UTF-8 顶层 JSON 对象（无额外字段），用于平台提交。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    return str(out)
