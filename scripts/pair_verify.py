# -*- coding: utf-8 -*-
"""闭合配对第一步：独立复核复制件结构 + 构造 A_o/A_s 锚（零成本，无新推理）。
复核点：① query↔gallery 字节级复制件（md5）；② 双向身份池 A_o/A_s 互异性；
③ 现役 prediction 的 top-1 落点与回环率；④ 锚资产落盘供后续证据矩阵复用。"""
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

WS = Path(r"h:\Ship-Re-Identification")
TT = WS / "question6-data" / "preliminary-round-test-data"
PROJ = WS / "Multi-source-Satellite-Cross-modal-Ship-Re-Identification-main"

task = json.loads((TT / "task.json").read_text(encoding="utf-8"))
qs, gs = task["queries"], task["gallery"]
print("数据目录:", sorted(p.name for p in TT.iterdir()))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


qmd5 = {q["query_id"]: md5(TT / q["image_path"]) for q in qs}
gmd5, g_by_md5 = {}, defaultdict(list)
for g in gs:
    m = md5(TT / g["image_path"])
    gmd5[g["image_id"]] = m
    g_by_md5[m].append(g["image_id"])

gmod = {g["image_id"]: g["modality"] for g in gs}
qtype = {q["query_id"]: q["query_type"] for q in qs}
o2s_ids = [q["query_id"] for q in qs if q["query_type"] == "O2S"]
s2o_ids = [q["query_id"] for q in qs if q["query_type"] == "S2O"]
o2o_ids = [q["query_id"] for q in qs if q["query_type"] == "O2O"]

stats, copy_src, multi = Counter(), {}, []
for q in qs:
    hits = g_by_md5.get(qmd5[q["query_id"]], [])
    if hits:
        stats[(q["query_type"], "exact")] += 1
        copy_src[q["query_id"]] = hits[0]
        if len(hits) > 1:
            multi.append((q["query_id"], hits))
    else:
        stats[(q["query_type"], "none")] += 1

print("[A] 精确复制件:", dict(stats))
print("    一场多图(同 md5 于 gallery 多行):", len(multi), multi[:3])
print("    用户引例: q_000001 ->", copy_src.get("q_000001"), "| q_000002 ->", copy_src.get("q_000002"))
badmod = [(q, gid) for q, gid in copy_src.items() if (gmod[gid] == "optical") != qtype[q].startswith("O")]
print("    复制件模态不一致:", len(badmod), badmod[:3])
rev = Counter(copy_src.values())
conf = [(k, v) for k, v in rev.items() if v > 1]
print("    复制源被多 query 共享:", len(conf), conf[:5])

# ---- feat 特征（7 成员）----
fd = PROJ / "sims" / "test_rerank_full"
members = {p.stem: torch.load(p, map_location="cpu") for p in sorted(fd.glob("feat_*.pt"))}
names = list(members)
ref = members[names[0]]
qids_f = [str(x) for x in ref["q_ids"]]
gids_f = [str(x) for x in ref["g_ids"]]
qi = {x: i for i, x in enumerate(qids_f)}
gi = {x: i for i, x in enumerate(gids_f)}
miss_q = [x for x in qtype if x not in qi]
miss_g = [g["image_id"] for g in gs if g["image_id"] not in gi]
print("[B] feat 文件 %d 个 | q 覆盖 %d（缺 %d）| g 覆盖 %d（缺 %d）" % (len(names), len(qi), len(miss_q), len(gi), len(miss_g)))

qfeats = [F.normalize(members[n]["q"].float(), dim=1) for n in names]
gfeats = [F.normalize(members[n]["g"].float(), dim=1) for n in names]
opt_ids = [g["image_id"] for g in gs if g["modality"] == "optical"]
sar_ids = [g["image_id"] for g in gs if g["modality"] == "sar"]
opt_cols = [gi[x] for x in opt_ids]
sar_cols = [gi[x] for x in sar_ids]
print("    gallery 模态: optical %d / sar %d" % (len(opt_cols), len(sar_cols)))


def same_mod_best(qid, cols, ids):
    s = None
    for qf, gf in zip(qfeats, gfeats):
        v = qf[qi[qid]] @ gf[cols].T
        s = v if s is None else s + v
    s = s / len(qfeats)
    top2 = torch.topk(s, 2).values
    k = int(torch.argmax(s))
    return ids[k], float(top2[0]), float(top2[1])


def build_anchor(ids_list, cols, ids, tag):
    anchor, near = {}, []
    for q in ids_list:
        if q in copy_src:
            anchor[q] = copy_src[q]
        else:
            gid, s1, s2 = same_mod_best(q, cols, ids)
            anchor[q] = gid
            near.append((q, gid, round(s1, 3), round(s2, 3)))
    print("[%s] 锚 %d 个（精确 %d / 近邻 %d），互异 %d" % (
        tag, len(anchor), len(anchor) - len(near), len(near), len(set(anchor.values()))))
    if near:
        s1s = sorted(x[2] for x in near)
        print("    近邻锚 s1 min/med/max = %.3f / %.3f / %.3f；样本:" % (s1s[0], s1s[len(s1s) // 2], s1s[-1]), near[:5])
    return anchor, near


A_o, o2s_near = build_anchor(o2s_ids, opt_cols, opt_ids, "C-o")
A_s, s2o_near = build_anchor(s2o_ids, sar_cols, sar_ids, "D-s")
A_o2o, o2o_near = build_anchor(o2o_ids, opt_cols, opt_ids, "E-o2o")

print("[E2] O2O: 精确复制 %d/27，近邻锚 sim 分布（应~0.79 典型）:",
      sum(1 for q in o2o_ids if q in copy_src), [x[2] for x in o2o_near][:8])

# ---- 与现役 prediction 交叉验证 ----
pred = json.loads((WS / "prediction.json").read_text(encoding="utf-8"))
A_s_set, A_o_set = set(A_s.values()), set(A_o.values())
c1 = sum(1 for q in o2s_ids if pred[q][0] in A_s_set)
c2 = sum(1 for q in s2o_ids if pred[q][0] in A_o_set)
c3 = sum(1 for q in o2o_ids if pred[q][0] in A_o_set)
print("[F] 现役 top-1 落点: O2S∈A_s %d/120 | S2O∈A_o %d/120 | O2O∈A_o %d/27" % (c1, c2, c3))
print("    随机基准: O2S %.3f | S2O %.3f（|A|/900）" % (len(A_s_set) / 900, len(A_o_set) / 900))

sar_back = {v: k for k, v in A_s.items()}  # SAR 锚 -> S2O query
opt_back = {v: k for k, v in A_o.items()}   # optical 锚 -> O2S query
loop_tot = loop_hit = 0
for q in o2s_ids:
    j = sar_back.get(pred[q][0])
    if j is not None:
        loop_tot += 1
        if pred[j][0] == A_o[q]:
            loop_hit += 1
print("[G] 回环: O2S top1∈A_s 的 %d 条中，配对 S2O 互指(闭环) %d 条（%.1f%%）"
      % (loop_tot, loop_hit, 100.0 * loop_hit / max(loop_tot, 1)))
tot = sum(1 for q in s2o_ids if pred[q][0] in opt_back)
print("    （镜像）S2O top1∈A_o 的 %d 条，配对 O2S 回指数 %d" % (c2, tot))

# ---- 落盘锚资产 ----
out = {
    "copy_src": copy_src,
    "A_o": A_o, "A_s": A_s, "A_o2o": A_o2o,
    "o2s_near": o2s_near, "s2o_near": s2o_near, "o2o_near": o2o_near,
}
op = Path(os.environ["TEMP"]) / "pair_anchors.json"
op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("[H] 锚资产已落盘:", op)