# -*- coding: utf-8 -*-
"""RRF（Reciprocal Rank Fusion）多模型预测融合，生成赛题提交 prediction.json。

score(c) = Σ_m w_m / (k + rank_m(c))，只使用各成员的 top-10 排名，不依赖相似度量纲，
因此可安全融合「特征直接 top-10」与「TTA+rerank 后的 top-10」等不同后处理协议。

依据（本地验证，1450 query）：
    A=E0 ep80 + ship_reid_vit + ep45 + ep25，等权 @k=10
    全量 0.6270 / fold0 0.6043 / fold1 0.6497（锚点 A 单模型 0.5888 / 0.5757 / 0.6019）

用法：
    python scripts/rrf_fuse.py \
        --members sims/test/pred_ep80.json sims/test/pred_ep45.json \
                  sims/test/pred_ep25.json ../ship_reid_vit/outputs/prediction.json \
        --weights 1 1 1 1 --k 10 \
        --task ../question6-data/preliminary-round-test-data/task.json \
        --out submission_rrf.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 查询方向 -> 目标 gallery 模态（赛题规则：O2S 检索 SAR，S2O/O2O 检索光学）
TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RRF ensemble of ReID prediction lists")
    p.add_argument("--members", nargs="+", required=True, help="成员预测 json（top-K 列表）")
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="各成员权重（默认全 1）")
    p.add_argument("--k", type=float, default=10.0, help="RRF 常数 k（默认 10）")
    p.add_argument("--topk", type=int, default=10, help="输出列表长度（默认 10）")
    p.add_argument("--task", type=str, default="", help="可选：task.json，用于校验输出合法性")
    p.add_argument("--out", type=str, required=True, help="输出 prediction.json 路径")
    return p.parse_args()


def load_pred(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rrf(members: list, weights: list, k: float, topk: int) -> dict:
    out = {}
    for qid in members[0]:
        s = {}
        for lst, w in zip(members, weights):
            for rank, iid in enumerate(lst.get(qid, []), 1):
                s[iid] = s.get(iid, 0.0) + w / (k + rank)
        out[qid] = [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:topk]]
    return out


def verify(pred: dict, task_path: str, topk: int) -> None:
    """按赛题平台判无效的条件逐条校验（缺/多 query、候选数≠10、ID 无效或重复、模态错误）。"""
    with open(task_path, encoding="utf-8") as f:
        task = json.load(f)
    q_ids = [q["query_id"] for q in task["queries"]]
    g_ids = {g["image_id"] for g in task["gallery"]}
    g_mod = {g["image_id"]: g["modality"] for g in task["gallery"]}
    assert set(pred) == set(q_ids), "query 集合与 task.json 不一致"
    n_mod_bad = 0
    for q in task["queries"]:
        qid, want = q["query_id"], TARGET_MOD.get(q["query_type"])
        if want is None:
            raise SystemExit("未知 query_type: {}".format(q["query_type"]))
        lst = pred[qid]
        assert len(lst) == topk, "{} 长度 {} != {}".format(qid, len(lst), topk)
        assert len(set(lst)) == topk, "{} 存在重复候选".format(qid)
        assert set(lst) <= g_ids, "{} 含非 gallery 候选".format(qid)
        n_mod_bad += sum(1 for gid in lst if g_mod[gid] != want)
    assert n_mod_bad == 0, "候选项模态错配 {} 个".format(n_mod_bad)
    print("校验通过: {} query / {} gallery / top-{}，候选合法且模态一致".format(
        len(q_ids), len(g_ids), topk))


def main() -> None:
    args = parse_args()
    members = [load_pred(m) for m in args.members]
    weights = args.weights if args.weights else [1.0] * len(members)
    if len(weights) != len(members):
        raise SystemExit("--weights 个数必须与 --members 一致")

    base = set(members[0])
    for m, path in zip(members[1:], args.members[1:]):
        if set(m) != base:
            raise SystemExit(f"query 集合不一致: {path}")

    pred = rrf(members, weights, args.k, args.topk)
    for path, w in zip(args.members, weights):
        print("  成员 w={:<4} {}".format(w, path))
    print("RRF 融合完成: {} query, k={}".format(len(pred), args.k))

    if args.task:
        verify(pred, args.task, args.topk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)
    print("已写入: {}".format(out))


if __name__ == "__main__":
    main()