"""推理：特征提取 + 余弦相似度检索 + Top-K 排序输出。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ShipReIDModel, build_model


class RetrievalEngine:
    """以一张查询图（或一批查询图）在候选图库中做余弦相似度检索。"""

    def __init__(self, cfg, ckpt_path: str, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt_path, map_location=self.device)
        # 用 checkpoint 内保存的训练配置修正架构关键字段（必须与训练一致才能正确加载权重）
        saved_cfg = state.get("cfg") if isinstance(state, dict) and "cfg" in state else None
        if isinstance(saved_cfg, dict):
            for key in ("backbone", "share_layer", "vit_split_layer", "num_classes", "image_size", "embedding_dim"):
                if saved_cfg.get(key) is not None:
                    setattr(cfg, key, saved_cfg[key])
        self.model = build_model(cfg).to(self.device)
        # 优先加载 EMA 权重（推理更稳定），回退到普通权重
        if isinstance(state, dict) and state.get("ema_state") is not None:
            self.model.load_state_dict(state["ema_state"])
        else:
            self.model.load_state_dict(state["model_state"] if "model_state" in state else state)
        self.model.eval()

    @torch.no_grad()
    def extract(self, loader: DataLoader, tta: bool = False) -> torch.Tensor:
        """提取特征。tta=True 时对原图与水平翻转图特征取平均。"""
        feats = []
        for imgs, _, modalities in loader:
            imgs = imgs.to(self.device)
            modalities = modalities.to(self.device)
            f = self.model.extract_feature(imgs, modalities)
            if tta:
                imgs_flip = torch.flip(imgs, dims=[3])  # 水平翻转
                f_flip = self.model.extract_feature(imgs_flip, modalities)
                f = (f + f_flip) / 2
            feats.append(f)
        return F.normalize(torch.cat(feats), dim=1)

    def retrieve(
        self, query_feats: torch.Tensor, gallery_feats: torch.Tensor, topk: int = 10
    ) -> torch.Tensor:
        """返回 (Q, topk) 的 gallery 索引排序（相似度降序）。"""
        sim = torch.matmul(query_feats, gallery_feats.t())  # (Q, G)
        return sim.topk(min(topk, gallery_feats.size(0)), dim=1).indices

    def retrieve_from_dataloaders(
        self, query_loader: DataLoader, gallery_loader: DataLoader, topk: int = 10
    ) -> torch.Tensor:
        q = self.extract(query_loader)
        g = self.extract(gallery_loader)
        return self.retrieve(q, g, topk)


def topk_results(
    indices: torch.Tensor,
    query_paths: List[str],
    gallery_paths: List[str],
    topk: int = 10,
) -> List[dict]:
    """把检索索引转为可读结果：[{query, candidates: [{path, rank, score}]}]。"""
    results = []
    for qi, row in enumerate(indices.tolist()):
        cands = []
        for rank, gi in enumerate(row[:topk], start=1):
            cands.append({"rank": rank, "path": gallery_paths[gi]})
        results.append({"query": query_paths[qi], "candidates": cands})
    return results


def save_results(results: List[dict], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path
