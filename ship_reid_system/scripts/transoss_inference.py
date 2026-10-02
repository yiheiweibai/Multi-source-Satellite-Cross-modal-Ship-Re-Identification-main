"""TransOSS 推理 + 后处理，生成赛题提交 prediction.json。

此脚本需在 TransOSS 仓库根目录运行（或 TransOSS 在 PYTHONPATH 中）。

用法（在 TransOSS 仓库根目录）：
    python transoss_inference.py \
        --config_file configs/hoss_transoss.yml \
        --weight weights/HOSS_TransOSS.pth \
        --task_json ../赛题6-初赛/初赛测试数据/task.json \
        --out_prediction prediction.json \
        --tta --rerank --qe --cluster

后处理与自研框架共用同一套算法（TTA/k-reciprocal/QE/Gallery聚类），
确保对比公平。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- TransOSS 依赖（需在 TransOSS 仓库内运行）----
from config import cfg
from model import make_model
from datasets.make_dataloader import make_dataloader
from processor import do_inference  # noqa: F401  (仅用于确认路径正确)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TransOSS inference + post-process for competition")
    p.add_argument("--config_file", default="configs/hoss_transoss.yml")
    p.add_argument("--weight", type=str, required=True, help="fine-tuned TransOSS weights")
    p.add_argument("--task_json", type=str, required=True, help="赛题 task.json")
    p.add_argument("--out_prediction", type=str, default="prediction.json")
    p.add_argument("--tta", action="store_true", help="水平翻转 TTA")
    p.add_argument("--rerank", action="store_true", help="k-reciprocal 重排序")
    p.add_argument("--qe", action="store_true", help="查询扩展")
    p.add_argument("--cluster", action="store_true", help="Gallery 聚类")
    p.add_argument("--cluster_eps", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_task(task_path: str):
    with open(task_path, encoding="utf-8") as f:
        task = json.load(f)
    base = Path(task_path).parent
    for q in task["queries"]:
        q["image_path_abs"] = str(base / q["image_path"])
    for g in task["gallery"]:
        g["image_path_abs"] = str(base / g["image_path"])
    return task


@torch.no_grad()
def extract_features(model, image_paths: list[str], modalities: list[int], cfg, device, tta: bool):
    """用 TransOSS 模型提取特征。modalities: 0=optical, 1=sar。"""
    from PIL import Image
    from torchvision import transforms as T
    from torch.utils.data import DataLoader, Dataset

    size = cfg.INPUT.SIZE_TEST  # [256, 128]
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    class ImgDataset(Dataset):
        def __init__(self, paths, mods):
            self.paths = paths
            self.mods = mods
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), self.mods[idx]

    loader = DataLoader(ImgDataset(image_paths, modalities), batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=4)
    feats = []
    model.eval()
    for imgs, mods in loader:
        imgs = imgs.to(device)
        camids = mods.to(device)
        # img_wh: 尺寸嵌入（SSE 模块需要）
        img_wh = torch.tensor([[size[1], size[0]]] * imgs.size(0), dtype=torch.float32).to(device)
        feat = model(imgs, cam_label=camids, img_wh=img_wh)
        if isinstance(feat, tuple):
            feat = feat[0]
        if tta:
            imgs_flip = torch.flip(imgs, dims=[3])
            feat_flip = model(imgs_flip, cam_label=camids, img_wh=img_wh)
            if isinstance(feat_flip, tuple):
                feat_flip = feat_flip[0]
            feat = (feat + feat_flip) / 2
        feats.append(feat.cpu())
    return F.normalize(torch.cat(feats), dim=1)


# ---- 后处理函数（与自研框架一致）----
def k_reciprocal_rerank(q, g, k1=20, k2=6, lambda_value=0.3):
    from sklearn.cluster import DBSCAN  # noqa
    import numpy as np
    Q, G = q.size(0), g.size(0)
    device = q.device
    sim_qg = torch.matmul(q, g.t())
    sim_gg = torch.matmul(g, g.t())
    k1 = min(k1, G)
    gg_topk = sim_gg.topk(k1, dim=1).indices
    V_g = torch.zeros(G, G, device=device)
    rows = torch.arange(G, device=device).unsqueeze(1).expand(-1, k1)
    V_g[rows, gg_topk] = 1.0
    qg_topk = sim_qg.topk(k1, dim=1).indices
    reranked = torch.zeros(Q, G, device=device)
    for qi in range(Q):
        cand = qg_topk[qi]
        V_q = torch.zeros(G, device=device)
        V_q[cand] = 1.0
        cand_V = V_g[cand]
        inter = (V_q.unsqueeze(0) * cand_V).sum(1)
        union = (V_q.unsqueeze(0) + cand_V).clamp(max=1).sum(1)
        reranked[qi, cand] = inter / union.clamp(min=1e-6)
    return lambda_value * reranked + (1 - lambda_value) * sim_qg


def query_expansion(q, g, topk=1, alpha=0.5):
    sim = torch.matmul(q, g.t())
    topk = min(topk, g.size(0))
    idx = sim.topk(topk, dim=1).indices
    avg = g[idx].mean(dim=1)
    return F.normalize(q + alpha * avg, dim=1)


def gallery_cluster(q, g, g_mods, eps=0.5, min_samples=2):
    from sklearn.cluster import DBSCAN
    import numpy as np
    G = g.size(0)
    center = g.clone()
    cids = torch.full((G,), -1, dtype=torch.long)
    for mod in (0, 1):
        idx = (g_mods == mod).nonzero(as_tuple=True)[0]
        if idx.numel() < min_samples:
            continue
        feats = g[idx].numpy()
        dist = np.clip(1.0 - feats @ feats.T, 0.0, None)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(dist).labels_
        cids[idx] = torch.as_tensor(labels, dtype=torch.long)
        for c in set(labels):
            if c == -1:
                continue
            ci = idx[labels == c]
            center[ci] = F.normalize(g[ci].mean(0), dim=0)
    sim_c = torch.matmul(q, center.t())
    sim_r = torch.matmul(q, g.t())
    noise = (cids == -1).unsqueeze(0).expand_as(sim_r)
    return torch.where(noise, sim_r, sim_c)


def main() -> None:
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    task = load_task(args.task_json)
    queries = task["queries"]
    gallery = task["gallery"]
    print(f"queries: {len(queries)}, gallery: {len(gallery)}")

    # 模型
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 用训练集类别数创建模型（加载权重时分类头会被忽略，只取特征提取部分）
    train_loader, _, _, _, num_query, num_classes, camera_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.load_param(args.weight)
    model = model.to(device)
    model.eval()
    print(f"模型加载完成: {args.weight}")

    MOD_ID = {"optical": 0, "sar": 1}
    QTYPE_TO_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}

    q_paths = [q["image_path_abs"] for q in queries]
    q_mods = [MOD_ID[QTYPE_TO_MOD[q["query_type"]]] for q in queries]
    g_paths = [g["image_path_abs"] for g in gallery]
    g_mods = [MOD_ID[g["modality"]] for g in gallery]
    g_mods_t = torch.tensor(g_mods, dtype=torch.long)

    print("提取 query 特征...")
    q_feats = extract_features(model, q_paths, q_mods, cfg, device, args.tta)
    print("提取 gallery 特征...")
    g_feats = extract_features(model, g_paths, g_mods, cfg, device, args.tta)
    print(f"q_feats: {q_feats.shape}, g_feats: {g_feats.shape}")

    if args.qe:
        q_feats = query_expansion(q_feats, g_feats)

    if args.cluster:
        sim = gallery_cluster(q_feats, g_feats, g_mods_t, eps=args.cluster_eps)
    elif args.rerank:
        sim = k_reciprocal_rerank(q_feats, g_feats)
    else:
        sim = torch.matmul(q_feats, g_feats.t())

    # 按 query_type 过滤候选模态
    prediction = {}
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        top = sim[qi, cand].topk(10).indices.tolist()
        prediction[q["query_id"]] = [gallery[cand[t]]["image_id"] for t in top]

    with open(args.out_prediction, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    print(f"prediction.json 已保存: {args.out_prediction}")
    print(f"后处理: TTA={args.tta}, rerank={args.rerank}, QE={args.qe}, cluster={args.cluster}")


if __name__ == "__main__":
    main()
