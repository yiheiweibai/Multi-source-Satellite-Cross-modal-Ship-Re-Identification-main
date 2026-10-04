"""SDF-Net 推理 + 后处理，生成赛题提交 prediction.json（本地验证协议）。

与 TransOSS 推理脚本（scripts/transoss_inference.py）保持同一套接口与后处理算法，
保证跨模型对比与融合公平：
  - 按 query_type 分组后处理（O2S 只在 SAR gallery 上 rerank/QE，S2O 只在 optical
    gallery 上，O2O 只在 optical gallery 且排除 query 自身图像）——与修复后的
    transoss_inference.py 完全一致；
  - SAR 图像保持三通道灰度复制形式（赛题硬约束），不做 GRAY2RGB 转换；
  - 支持 --save_sim 导出 q×g 相似度矩阵与元数据，供跨模型融合脚本使用。

用法（在 SDF-Net 仓库根目录运行）：
    python ..\\scripts\\sdfnet_inference.py \
        --config_file configs/SDF-Net.yml \
        --weight logs/SDF-Net/SDF-Net.pth \
        --task_json ..\\..\\赛题6-初赛\\训练数据\\local_val_task.json \
        --out_prediction prediction_sdfnet.json \
        --save_sim sims/sdfnet

可选：--tta / --rerank / --qe（默认 base 纯特征）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- SDF-Net 依赖（需在 SDF-Net 仓库根目录运行）----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SDF-Net"))
from config import cfg  # noqa: E402
from model import make_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDF-Net inference + post-process for competition")
    p.add_argument("--config_file", default="configs/SDF-Net.yml")
    p.add_argument("--weight", type=str, required=True, help="SDF-Net weights (official or fine-tuned)")
    p.add_argument("--task_json", type=str, required=True, help="赛题 task.json")
    p.add_argument("--out_prediction", type=str, default="prediction_sdfnet.json")
    p.add_argument("--save_sim", type=str, default="", help="导出 sim.npy/meta.json 的目录（跨模型融合用）")
    p.add_argument("--tta", action="store_true", help="水平翻转 TTA")
    p.add_argument("--rerank", action="store_true", help="k-reciprocal 重排序")
    p.add_argument("--qe", action="store_true", help="查询扩展")
    p.add_argument("--batch_size", type=int, default=8, help="特征提取 batch size（16GB 显存建议 8~32）")
    p.add_argument("--num_workers", type=int, default=4)
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
def extract_features(model, image_paths: list[str], modalities: list[int], cfg, device,
                     tta: bool, batch_size: int, num_workers: int):
    """用 SDF-Net 提取特征。modalities: 0=optical, 1=sar。

    与 TransOSS 推理一致：PIL 读图 convert("RGB")，SAR 三通道灰度复制原样保留；
    img_wh 使用与 transoss_inference.py 相同的简化尺寸嵌入 [size[1], size[0]]。
    """
    from PIL import Image
    from torchvision import transforms as T
    from torch.utils.data import DataLoader, Dataset

    size = cfg.INPUT.SIZE_TEST  # [256, 128]（SDF-Net.yml 默认，与官方配置一致）
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
            # SAR 三通道灰度复制形式原样保留（赛题硬约束），统一 convert("RGB")
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), self.mods[idx]

    loader = DataLoader(ImgDataset(image_paths, modalities), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats = []
    model.eval()
    for imgs, mods in loader:
        imgs = imgs.to(device)
        camids = mods.to(device)
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


# ---- 后处理函数（与 transoss_inference.py 完全一致，保证对比公平）----
def k_reciprocal_rerank(q, g, k1=20, k2=6, lambda_value=0.3):
    import numpy as np  # noqa: F401
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


def main() -> None:
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    task = load_task(args.task_json)
    queries = task["queries"]
    gallery = task["gallery"]
    print(f"queries: {len(queries)}, gallery: {len(gallery)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 与官方 test.py 一致：num_class=0（classifier 权重在 load_param 中被跳过）
    model = make_model(cfg, num_class=0, camera_num=2)
    model.load_param(args.weight)
    model = model.to(device)
    model.eval()
    print(f"模型加载完成: {args.weight}")

    MOD_ID = {"optical": 0, "sar": 1}
    QTYPE_TO_QUERY_MOD = {"O2S": "optical", "S2O": "sar", "O2O": "optical"}
    QTYPE_TO_TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}

    q_paths = [q["image_path_abs"] for q in queries]
    q_mods = [MOD_ID[QTYPE_TO_QUERY_MOD[q["query_type"]]] for q in queries]
    g_paths = [g["image_path_abs"] for g in gallery]
    g_mods = [MOD_ID[g["modality"]] for g in gallery]

    print("提取 query 特征...")
    q_feats = extract_features(model, q_paths, q_mods, cfg, device, args.tta,
                               args.batch_size, args.num_workers)
    print("提取 gallery 特征...")
    g_feats = extract_features(model, g_paths, g_mods, cfg, device, args.tta,
                               args.batch_size, args.num_workers)
    print(f"q_feats: {q_feats.shape}, g_feats: {g_feats.shape}")

    # 按 query_type 分组后处理（与修复后的 transoss_inference.py 同逻辑）：
    # rerank/QE 仅在目标候选模态子集内进行；O2O 额外排除 query 自身图像。
    sim = torch.zeros(q_feats.size(0), g_feats.size(0), dtype=torch.float32)
    g_abs = [str(Path(g["image_path_abs"]).resolve()) for g in gallery]
    q_abs = [str(Path(q["image_path_abs"]).resolve()) for q in queries]
    for qtype, target_mod in QTYPE_TO_TARGET_MOD.items():
        q_idx = [i for i, q in enumerate(queries) if q["query_type"] == qtype]
        if not q_idx:
            continue
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        if qtype == "O2O":
            own = {q_abs[j] for j in q_idx}
            cand = [i for i in cand if g_abs[i] not in own]
        if len(cand) == 0:
            continue
        cand_t = torch.tensor(cand, dtype=torch.long)
        g_mod_feats = g_feats[cand_t]
        q_work = q_feats[torch.as_tensor(q_idx, dtype=torch.long)]
        if args.qe:
            q_work = query_expansion(q_work, g_mod_feats)
        if args.rerank:
            sim[torch.as_tensor(q_idx, dtype=torch.long)[:, None], cand_t] = k_reciprocal_rerank(q_work, g_mod_feats)
        else:
            sim[torch.as_tensor(q_idx, dtype=torch.long)[:, None], cand_t] = torch.matmul(q_work, g_mod_feats.t())

    # 保存相似度矩阵 + 元数据（跨模型融合用）
    if args.save_sim:
        out_dir = Path(args.save_sim)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "queries": [{"query_id": q["query_id"], "query_type": q["query_type"],
                         "image_path": q["image_path"]} for q in queries],
            "gallery": [{"image_id": g["image_id"], "modality": g["modality"],
                         "image_path": g["image_path"]} for g in gallery],
        }
        torch.save(sim, out_dir / "sim.pt")
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"sim/meta 已保存: {out_dir}")

    # 按 query_type 过滤候选模态，并排除 query 自身图像（防御本地验证协议重叠）
    prediction = {}
    for qi, q in enumerate(queries):
        target_mod = QTYPE_TO_TARGET_MOD[q["query_type"]]
        cand = [i for i, g in enumerate(gallery) if g["modality"] == target_mod]
        q_abs_self = str(Path(q["image_path_abs"]).resolve())
        cand = [i for i in cand if g_abs[i] != q_abs_self]
        if len(cand) < 10:
            raise RuntimeError(f"query {q['query_id']} 候选仅 {len(cand)} 个，不足 10")
        top = sim[qi, torch.tensor(cand)].topk(10).indices.tolist()
        prediction[q["query_id"]] = [gallery[cand[t]]["image_id"] for t in top]

    with open(args.out_prediction, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    print(f"prediction.json 已保存: {args.out_prediction}")
    print(f"后处理: TTA={args.tta}, rerank={args.rerank}, QE={args.qe}")


if __name__ == "__main__":
    main()
