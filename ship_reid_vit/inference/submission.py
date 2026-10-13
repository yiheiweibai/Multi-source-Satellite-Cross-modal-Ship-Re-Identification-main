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
from inference.postprocess import cluster_retrieval, k_reciprocal_rerank, query_expansion
from inference.retrieval import RetrievalEngine


@torch.no_grad()
def compute_sim(
    engine: RetrievalEngine,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    queries: List[Dict],
    gallery: List[Dict],
    tta: bool = False,
    rerank: bool = False,
    qe: bool = False,
    cluster: bool = False,
    multiscale: bool = False,
    ms_weights: tuple = (0.5, 0.3, 0.2),
    directional: bool = False,
):
    """提取特征并计算 (Q, G) 相似度矩阵（提交链路与产物落盘共用）。

    Args:
        directional: 方向特异性投影（Phase 4），query 与其目标 gallery 在同一投影空间内
            计算相似度（O2S 同过 proj_o2s、S2O 同过 proj_s2o），忽略 qe/rerank/cluster

    Returns:
        (sim, q_feats, g_feats, g_mods)
    """
    q_feats = engine.extract(
        query_loader, tta=tta, multiscale=multiscale, ms_weights=ms_weights,
    )   # (Q, D)
    g_feats = engine.extract(
        gallery_loader, tta=tta, multiscale=multiscale, ms_weights=ms_weights,
    )  # (G, D)

    # 设备统一到特征所在设备
    dev = q_feats.device
    g_mods = torch.as_tensor(
        [MODALITY_ID[g["modality"]] for g in gallery], dtype=torch.long, device=dev
    )

    # Phase 4：方向特异性投影相似度（query 与目标 gallery 必须同空间投影）
    if directional and getattr(engine.model, "use_directional", False):
        sim = _directional_sim(engine, q_feats, g_feats, g_mods, queries)
        return sim, q_feats, g_feats, g_mods

    # 后处理一律按目标候选模态分组执行
    sim = torch.zeros(q_feats.size(0), g_feats.size(0), device=dev, dtype=torch.float32)
    for mod_id in (0, 1):
        mask = g_mods == mod_id
        cand = mask.nonzero().squeeze(1)
        if cand.numel() == 0:
            continue
        g_mod_feats = g_feats[cand]                     # (G_m, D)
        q_work = q_feats
        if qe:
            q_work = query_expansion(q_work, g_mod_feats, topk=1, alpha=0.5)
        if cluster:
            sim[:, cand] = cluster_retrieval(
                q_work, g_mod_feats, g_mods[mask], eps=0.5, min_samples=2,
            )
        elif rerank:
            sim[:, cand] = k_reciprocal_rerank(q_work, g_mod_feats, k1=20, k2=6, lambda_value=0.3)
        else:
            sim[:, cand] = torch.matmul(q_work, g_mod_feats.t())
    return sim, q_feats, g_feats, g_mods


@torch.no_grad()
def _directional_sim(
    engine: RetrievalEngine,
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    g_mods: torch.Tensor,
    queries: List[Dict],
) -> torch.Tensor:
    """方向特异性相似度（Phase 4）。

    O2S query 与其目标 SAR gallery 同过 proj_o2s 投影后计算相似度；
    S2O query 与其目标光学 gallery 同过 proj_s2o 投影；O2O 用标准特征。
    """
    dh = engine.model.directional_head
    dev = q_feats.device
    q_types = [q["query_type"] for q in queries]

    # query 侧按方向投影
    q_work = q_feats.clone()
    for qt in ("O2S", "S2O"):
        rows = [i for i, t in enumerate(q_types) if t == qt]
        if not rows:
            continue
        rt = torch.tensor(rows, dtype=torch.long, device=dev)
        q_work[rt] = dh.extract_directional(q_feats[rt], qt)
    q_work = F.normalize(q_work, dim=1)

    # gallery 侧按目标模态投影（O2S 目标=SAR，S2O 目标=光学）
    sar_idx = (g_mods == 1).nonzero().squeeze(1)
    opt_idx = (g_mods == 0).nonzero().squeeze(1)
    g_o2s = (
        F.normalize(dh.extract_directional(g_feats[sar_idx], "O2S"), dim=1)
        if sar_idx.numel() else None
    )
    g_s2o = (
        F.normalize(dh.extract_directional(g_feats[opt_idx], "S2O"), dim=1)
        if opt_idx.numel() else None
    )

    sim = torch.zeros(q_work.size(0), g_feats.size(0), device=dev, dtype=torch.float32)
    for qt, cols, gmat in (("O2S", sar_idx, g_o2s), ("S2O", opt_idx, g_s2o)):
        rows = [i for i, t in enumerate(q_types) if t == qt]
        if not rows or cols.numel() == 0 or gmat is None:
            continue
        rt = torch.tensor(rows, dtype=torch.long, device=dev)
        sim[rt.unsqueeze(1), cols.unsqueeze(0)] = q_work[rt] @ gmat.t()
    # O2O（及兜底）：标准特征空间
    o2o_rows = [i for i, t in enumerate(q_types) if t == "O2O"]
    if o2o_rows:
        rt = torch.tensor(o2o_rows, dtype=torch.long, device=dev)
        sim[rt] = q_work[rt] @ g_feats.t()
    return sim


def indices_from_sim(
    sim: torch.Tensor,
    g_mods: torch.Tensor,
    queries: List[Dict],
    gallery: List[Dict],
    topk: int = 10,
) -> torch.Tensor:
    """由相似度矩阵取每条 query 的 (topk,) gallery 全局索引（强制模态过滤 + 排除自匹配）。"""
    dev = sim.device
    indices = torch.full((sim.size(0), topk), -1, dtype=torch.long, device=dev)
    # gallery 绝对路径（用于排除 query 自身图像；官方 task 无重叠，本地验证协议可能重叠）
    g_paths = [str(Path(g.get("image_path_abs") or g["image_path"]).resolve()) for g in gallery]
    for qi, query in enumerate(queries):
        target_mod = QUERY_TYPE_TO_MODALITY[query["query_type"]]
        mask = g_mods == MODALITY_ID[target_mod]
        cand = mask.nonzero().squeeze(1)
        q_path = str(Path(query.get("image_path_abs") or query["image_path"]).resolve())
        keep = torch.tensor(
            [gp != q_path for gp in (g_paths[i] for i in cand.tolist())],
            dtype=torch.bool, device=dev,
        )
        cand = cand[keep]
        if cand.numel() < topk:
            raise RuntimeError(
                f"query {query['query_id']} 目标模态候选仅 {cand.numel()} 个，不足 {topk}"
            )
        top = sim[qi, cand].topk(topk).indices
        indices[qi] = cand[top]
    return indices


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
    cluster: bool = False,
    multiscale: bool = False,
    ms_weights: tuple = (0.5, 0.3, 0.2),
    directional: bool = False,
) -> torch.Tensor:
    """检索并强制候选模态过滤，返回 (Q, topk) 的 gallery 全局索引（相似度降序）。

    Args:
        tta: 水平翻转 TTA（原图+翻转特征平均）
        rerank: k-reciprocal 重排序
        qe: 查询扩展（用 top-1 gallery 特征扩展 query）
        cluster: Gallery 聚类 + 类别中心检索
        multiscale: 多尺度特征融合（全局+中心局部+水平条带）
        ms_weights: 多尺度权重 (global, center, stripe)
        directional: 方向特异性投影（Phase 4）
    """
    sim, _, _, g_mods = compute_sim(
        engine, query_loader, gallery_loader, queries, gallery,
        tta=tta, rerank=rerank, qe=qe, cluster=cluster,
        multiscale=multiscale, ms_weights=ms_weights,
        directional=directional,
    )
    return indices_from_sim(sim, g_mods, queries, gallery, topk=topk)


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
    cluster: bool = False,
    multiscale: bool = False,
    ms_weights: tuple = (0.5, 0.3, 0.2),
    directional: bool = False,
) -> Tuple[Dict[str, List[str]], List[Dict], List[Dict]]:
    """生成提交字典 {query_id: [10 个 gallery image_id]}。"""
    query_loader, gallery_loader, queries, gallery = build_test_loaders(
        cfg, task_json_path, max_queries=max_queries, max_gallery=max_gallery
    )
    engine = RetrievalEngine(cfg, ckpt_path, device=device)
    indices = retrieve_for_submission(
        engine, query_loader, gallery_loader, queries, gallery, topk,
        tta=tta, rerank=rerank, qe=qe, cluster=cluster,
        multiscale=multiscale, ms_weights=ms_weights,
        directional=directional,
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
