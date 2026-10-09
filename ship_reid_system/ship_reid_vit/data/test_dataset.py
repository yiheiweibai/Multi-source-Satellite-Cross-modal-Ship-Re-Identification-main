"""初赛/复赛测试数据加载：解析 task.json 并构建 Query / Gallery 数据集。

task.json 顶层固定包含 queries 与 gallery：
    queries: [{query_id, image_path, query_type}]，query_type ∈ {O2S, S2O, O2O}
    gallery: [{image_id, image_path, modality}]，modality ∈ {optical, sar}

候选模态约束（赛题）：O2S -> sar，S2O -> optical，O2O -> optical。
query 自身模态由 query_type 推断：O2S/O2O 的 query 为 optical，S2O 的 query 为 sar。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .dataset import MODALITY_ID

# query_type -> 目标候选模态（检索时必须过滤 Gallery 为对应模态）
QUERY_TYPE_TO_MODALITY = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}
# query_type -> query 自身模态（用于双分支骨干选择）
QUERY_TYPE_TO_QUERY_MODALITY = {"O2S": "optical", "S2O": "sar", "O2O": "optical"}


def load_task(task_json_path: str) -> Dict:
    """读取测试包 task.json，返回 {queries: [...], gallery: [...]}。"""
    path = Path(task_json_path)
    if not path.exists():
        raise FileNotFoundError(f"task.json 不存在: {task_json_path}")
    with open(path, "r", encoding="utf-8") as f:
        task = json.load(f)
    if "queries" not in task or "gallery" not in task:
        raise ValueError(f"task.json 必须包含 queries 与 gallery 字段: {task_json_path}")
    base = path.parent
    for q in task["queries"]:
        if q["query_type"] not in QUERY_TYPE_TO_MODALITY:
            raise ValueError(f"未知 query_type: {q.get('query_type')}（{q.get('query_id')}）")
        q["image_path_abs"] = str((base / q["image_path"]).resolve())
    for g in task["gallery"]:
        if g["modality"] not in MODALITY_ID:
            raise ValueError(f"未知 gallery 模态: {g.get('modality')}（{g.get('image_id')}）")
        g["image_path_abs"] = str((base / g["image_path"]).resolve())
    return task


class QueryDataset(Dataset):
    """查询集。__getitem__ 返回 (img, query_id, query_modality_id)，
    便于 RetrievalEngine.extract 复用 (imgs, _, modalities) 契约。
    """

    def __init__(
        self,
        queries: List[Dict],
        transform: Optional[Callable] = None,
    ) -> None:
        self.queries = queries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        q = self.queries[idx]
        img = Image.open(q["image_path_abs"]).convert("RGB")
        mod = MODALITY_ID[QUERY_TYPE_TO_QUERY_MODALITY[q["query_type"]]]
        if self.transform is not None:
            img = self.transform(img, mod)
        return img, q["query_id"], mod

    def query_types(self) -> List[str]:
        return [q["query_type"] for q in self.queries]


class GalleryDataset(Dataset):
    """候选图库。__getitem__ 返回 (img, image_id, modality_id)。"""

    def __init__(
        self,
        gallery: List[Dict],
        transform: Optional[Callable] = None,
    ) -> None:
        self.gallery = gallery
        self.transform = transform

    def __len__(self) -> int:
        return len(self.gallery)

    def __getitem__(self, idx: int):
        g = self.gallery[idx]
        img = Image.open(g["image_path_abs"]).convert("RGB")
        mod = MODALITY_ID[g["modality"]]
        if self.transform is not None:
            img = self.transform(img, mod)
        return img, g["image_id"], mod

    def modalities(self) -> List[int]:
        return [MODALITY_ID[g["modality"]] for g in self.gallery]


def build_test_loaders(cfg, task_json_path: str, max_queries: int = 0, max_gallery: int = 0):
    """便捷函数：加载 task.json 并返回 (query_loader, gallery_loader, queries, gallery)。

    max_queries / max_gallery > 0 时只取前 N 条（冒烟/调试用）。
    """
    from torch.utils.data import DataLoader

    from .transforms import build_transforms

    task = load_task(task_json_path)
    queries = task["queries"]
    gallery = task["gallery"]
    if max_queries and max_queries > 0:
        queries = queries[:max_queries]
    if max_gallery and max_gallery > 0:
        gallery = gallery[:max_gallery]

    transform = build_transforms(cfg, is_train=False)
    query_ds = QueryDataset(queries, transform=transform)
    gallery_ds = GalleryDataset(gallery, transform=transform)
    num_workers = getattr(cfg, "num_workers", 0)

    query_loader = DataLoader(
        query_ds, batch_size=getattr(cfg, "test_batch_size", 32),
        shuffle=False, num_workers=num_workers,
    )
    gallery_loader = DataLoader(
        gallery_ds, batch_size=getattr(cfg, "test_batch_size", 32),
        shuffle=False, num_workers=num_workers,
    )
    return query_loader, gallery_loader, queries, gallery
