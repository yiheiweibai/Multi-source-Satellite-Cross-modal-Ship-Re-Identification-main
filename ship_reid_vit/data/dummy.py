"""模拟数据生成器。

数据开放前用于跑通全流程：按"船体模板 + 光学/SAR 渲染"生成带身份标注的
模拟图像，并写入 data/dummy/ 与 data/annotations/{train,val}.json。
真实数据开放后设置 config.dummy_mode=False 并指向真实标注即可。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

IMG_SIZE = 128
PER_CLASS = 8          # 每身份每种模态 8 张
TRAIN_RATIO = 0.8


def _make_ship_template(rng: random.Random, cls_id: int):
    """生成一艘"船"的结构模板：船体长宽、上层建筑块数/位置/尺寸、烟囱标志。"""
    hull_w = rng.randint(60, 110)
    hull_h = rng.randint(18, 30)
    n_super = rng.randint(1, 3)
    super_blocks = []
    for _ in range(n_super):
        bw = rng.randint(14, 30)
        bh = rng.randint(10, 20)
        bx = rng.randint(6, max(7, hull_w - bw - 6))
        super_blocks.append((bx, bw, bh))
    chimney = rng.randint(0, 1) == 1
    chimney_x = rng.randint(10, max(11, hull_w - 10))
    return {
        "hull_w": hull_w,
        "hull_h": hull_h,
        "super_blocks": super_blocks,
        "chimney": chimney,
        "chimney_x": chimney_x,
        "seed": cls_id,
    }


def _render_optical(tpl, rng: random.Random) -> np.ndarray:
    """光学渲染：海面背景 + 灰度船体 + 上层建筑 + 阴影 + 光照噪声。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    sea = 60 + 30 * rng.random()
    img[:, :] = [sea, sea + 8, sea + 16]  # 偏蓝海面
    # 光照渐变
    grad = np.linspace(0.85, 1.15, IMG_SIZE)[None, :, None]
    img *= grad
    x0 = (IMG_SIZE - tpl["hull_w"]) // 2 + rng.randint(-4, 4)
    y0 = (IMG_SIZE - tpl["hull_h"]) // 2 + rng.randint(-4, 4)
    hull_gray = 90 + rng.randint(0, 40)
    img[y0 : y0 + tpl["hull_h"], x0 : x0 + tpl["hull_w"]] = hull_gray
    # 上层建筑（稍亮）
    for (bx, bw, bh) in tpl["super_blocks"]:
        img[y0 - bh : y0, x0 + bx : x0 + bx + bw] = hull_gray + 45
    # 烟囱
    if tpl["chimney"]:
        img[y0 - 12 : y0, x0 + tpl["chimney_x"] : x0 + tpl["chimney_x"] + 5] = 30
    # 阴影（船下侧）
    img[y0 + tpl["hull_h"] : y0 + tpl["hull_h"] + 8, x0 + 4 : x0 + tpl["hull_w"] - 4] *= 0.6
    # 高斯噪声
    img += np.random.normal(0, 6, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _render_sar(tpl, rng: random.Random) -> np.ndarray:
    """SAR 渲染：暗背景 + 强散射船体 + 边缘亮线 + 乘性斑点噪声。"""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 12.0, dtype=np.float32)
    x0 = (IMG_SIZE - tpl["hull_w"]) // 2 + rng.randint(-4, 4)
    y0 = (IMG_SIZE - tpl["hull_h"]) // 2 + rng.randint(-4, 4)
    # 船体强散射
    img[y0 : y0 + tpl["hull_h"], x0 : x0 + tpl["hull_w"]] = 140 + 40 * rng.random()
    # 边缘亮线
    img[y0, x0 : x0 + tpl["hull_w"]] = 220
    img[y0 + tpl["hull_h"] - 1, x0 : x0 + tpl["hull_w"]] = 220
    # 上层建筑角反射（更强散射点）
    for (bx, bw, bh) in tpl["super_blocks"]:
        img[y0 - bh : y0, x0 + bx : x0 + bx + bw] = 170 + 40 * rng.random()
        img[y0 - bh, x0 + bx : x0 + bx + bw] = 235
    if tpl["chimney"]:
        img[y0 - 12 : y0, x0 + tpl["chimney_x"] : x0 + tpl["chimney_x"] + 5] = 200
    # 乘性相干斑
    l_equiv = 4.0
    speckle = np.random.gamma(l_equiv, 1.0 / l_equiv, img.shape).astype(np.float32)
    img *= speckle
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_dummy_data(
    data_root: str = "data",
    num_classes: int = 40,
    per_class: int = PER_CLASS,
    seed: int = 0,
) -> List[str]:
    """生成模拟数据与标注，返回生成的标注文件路径列表。"""
    root = Path(data_root)
    dummy_dir = root / "dummy"
    ann_dir = root / "annotations"
    rng = random.Random(seed)

    train_ann: List[dict] = []
    val_ann: List[dict] = []
    n_train_classes = int(num_classes * TRAIN_RATIO)

    for cls_id in range(num_classes):
        tpl = _make_ship_template(rng, cls_id)
        is_train = cls_id < n_train_classes
        for j in range(per_class):
            for mod, render_fn in (("optical", _render_optical), ("sar", _render_sar)):
                img = render_fn(tpl, rng)
                save_dir = dummy_dir / mod / f"ship_{cls_id:04d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{j:02d}.png"
                Image.fromarray(img).save(save_path)
                item = {"image_path": str(save_path.resolve()), "identity": cls_id, "modality": mod}
                (train_ann if is_train else val_ann).append(item)

    ann_dir.mkdir(parents=True, exist_ok=True)
    train_path = ann_dir / "train.json"
    val_path = ann_dir / "val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_ann, f, ensure_ascii=False, indent=1)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_ann, f, ensure_ascii=False, indent=1)
    return [str(train_path), str(val_path)]


def ensure_dummy_data(cfg) -> List[str]:
    """若标注缺失则生成模拟数据（幂等）。"""
    train_path = Path(cfg.train_ann)
    if train_path.exists():
        return [cfg.train_ann, cfg.val_ann]
    return generate_dummy_data(cfg.data_root, cfg.num_classes, PER_CLASS, cfg.seed)
