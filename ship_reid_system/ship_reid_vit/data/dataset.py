"""数据集接口：加载光学/SAR 舰船图像。

标注文件约定（JSON）：
    [
        {"image_path": "data/raw/optical/ship_001.jpg", "identity": 0, "modality": "optical"},
        {"image_path": "data/raw/sar/ship_001.png",    "identity": 0, "modality": "sar"}
    ]
- identity: 整数身份 ID（同一艘船 = 同一 ID）
- modality: "optical" 或 "sar"；缺省时按路径关键字自动推断
- 数据开放后把标注文件路径填入 config.train_ann / val_ann 即可使用

官方 labels.csv 支持：四列 image_id,ship_id,modality,image_path。
- 通过 load_labels_csv() 读取并归一化，ShipReIDCSVDataset 直接消费；
- image_path 为相对 csv 所在数据包根目录的相对路径；
- ship_id 为字符串（如 id_000001），自动映射为连续整数身份。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

MODALITY_ID = {"optical": 0, "sar": 1}
MODALITY_NAME = {0: "optical", 1: "sar"}


def _infer_modality(path: str) -> str:
    """按路径关键字推断模态，优先级：annotation 字段 > 路径关键字。"""
    p = path.lower()
    if "sar" in p or "synthetic" in p:
        return "sar"
    if "optical" in p or "opt" in p or "visible" in p or "rgb" in p:
        return "optical"
    # 目录名猜测：data/.../optical/... 或 data/.../sar/...
    parts = Path(path).parts
    for part in reversed(parts):
        lp = part.lower()
        if "sar" in lp:
            return "sar"
        if "opt" in lp or "visible" in lp:
            return "optical"
    raise ValueError(f"无法从路径推断模态，请在标注中显式给出 modality 字段: {path}")


def load_annotation(ann_path: str) -> List[Dict]:
    """读取标注 json，返回归一化样本列表。"""
    path = Path(ann_path)
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {ann_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples: List[Dict] = []
    for item in raw:
        img_path = str(item.get("image_path") or item.get("path"))
        if not img_path:
            raise ValueError(f"标注条目缺少 image_path: {item}")
        identity = int(item["identity"])
        modality = item.get("modality") or _infer_modality(img_path)
        modality = modality.lower()
        if modality not in MODALITY_ID:
            raise ValueError(f"未知模态: {modality}，应为 optical/sar")
        samples.append(
            {"image_path": img_path, "identity": identity, "modality": MODALITY_ID[modality]}
        )
    return samples


def load_labels_csv(csv_path: str, data_root: Optional[str] = None) -> List[Dict]:
    """读取官方 labels.csv（UTF-8，四列），返回归一化样本列表。

    返回每项：
        {image_path: 绝对路径, identity: 连续整数, modality: int(0/1),
         image_id: 原始 image_id, ship_id: 原始字符串}
    ship_id 按首次出现顺序映射为 0..N-1，保证同船同 ID。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"labels.csv 不存在: {csv_path}")
    root = Path(data_root) if data_root else path.parent
    raw_rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "ship_id", "modality", "image_path"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"labels.csv 列名不符，需要 {sorted(required)}，实际 {reader.fieldnames}"
            )
        for row in reader:
            modality = row["modality"].strip().lower()
            if modality not in MODALITY_ID:
                raise ValueError(f"未知模态 {modality!r}（行 {row.get('image_id', '?')}）")
            img_rel = row["image_path"].strip().replace("/", "\\")
            raw_rows.append(
                {
                    "image_id": row["image_id"].strip(),
                    "ship_id": row["ship_id"].strip(),
                    "modality": MODALITY_ID[modality],
                    "image_path": str((root / img_rel).resolve()),
                }
            )
    # 身份映射（稳定有序，与 ship_id 字符串语义一致）
    ship_ids = [r["ship_id"] for r in raw_rows]
    uniq_ids: List[str] = []
    seen = set()
    for sid in ship_ids:
        if sid not in seen:
            seen.add(sid)
            uniq_ids.append(sid)
    id_map = {sid: i for i, sid in enumerate(uniq_ids)}
    for r in raw_rows:
        r["identity"] = id_map[r["ship_id"]]
    return raw_rows


class ShipReIDDataset(Dataset):
    """光学/SAR 舰船重识别数据集。

    Args:
        ann_path: 标注 json 路径
        transform: 通用图像变换（PIL -> Tensor）
        sar_speckle: 仅对 SAR 样本施加的增强（斑点模拟等），训练时启用
        is_train: 是否训练集（决定是否施加 SAR 专属增强）
    """

    def __init__(
        self,
        ann_path: str,
        transform: Optional[Callable] = None,
        sar_speckle: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        self.samples = load_annotation(ann_path)
        self.transform = transform
        self.sar_speckle = sar_speckle
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img, s["modality"])
        if self.is_train and self.sar_speckle is not None and s["modality"] == 1:
            img = self.sar_speckle(img)
        return img, s["identity"], s["modality"]

    def labels(self) -> np.ndarray:
        return np.asarray([s["identity"] for s in self.samples], dtype=np.int64)

    def modalities(self) -> np.ndarray:
        return np.asarray([s["modality"] for s in self.samples], dtype=np.int64)

    def paths(self) -> List[str]:
        return [s["image_path"] for s in self.samples]

    def num_classes(self) -> int:
        return int(np.unique(self.labels()).size)


class ShipReIDCSVDataset(Dataset):
    """官方 labels.csv 数据集（image_id,ship_id,modality,image_path）。

    与 ShipReIDDataset 同接口，另提供 num_classes() / image_ids()。
    max_samples > 0 时仅取前 N 个样本（冒烟/调试用）。
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        sar_speckle: Optional[Callable] = None,
        is_train: bool = True,
        max_samples: int = 0,
    ) -> None:
        samples = load_labels_csv(csv_path)
        if max_samples and max_samples > 0:
            samples = samples[:max_samples]
        self.samples = samples
        self.transform = transform
        self.sar_speckle = sar_speckle
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img, s["modality"])
        if self.is_train and self.sar_speckle is not None and s["modality"] == 1:
            img = self.sar_speckle(img)
        return img, s["identity"], s["modality"]

    def labels(self) -> np.ndarray:
        return np.asarray([s["identity"] for s in self.samples], dtype=np.int64)

    def modalities(self) -> np.ndarray:
        return np.asarray([s["modality"] for s in self.samples], dtype=np.int64)

    def paths(self) -> List[str]:
        return [s["image_path"] for s in self.samples]

    def image_ids(self) -> List[str]:
        return [s["image_id"] for s in self.samples]

    def num_classes(self) -> int:
        return int(np.unique(self.labels()).size)
