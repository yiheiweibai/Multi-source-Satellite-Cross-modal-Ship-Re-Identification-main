# -*- coding: utf-8 -*-
"""赛题提交文件独立校验器（不依赖融合脚本，可校验任意来源的 prediction.json）。

python scripts/check_prediction.py --prediction prediction.json --task task.json

逐条对照官方规则的「直接判定无效」条件：
    文件名/编码、顶层结构、query 集合完全一致、每条恰好 topk 个候选、
    候选互不重复且均为合法 gallery id、候选模态与 query 方向一致、
    数组内不得出现相似度/路径/模态等额外字段。
额外再查一项工程泄漏：候选里不得出现 query 自身的图像。

任一失败 -> 退出码 1（用显式报错，不用 assert，避免 python -O 下被剥离）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# 查询方向 -> 目标 gallery 模态（赛题规则：O2S 检索 SAR，S2O/O2O 检索光学）
TARGET_MOD = {"O2S": "sar", "S2O": "optical", "O2O": "optical"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a competition prediction.json")
    p.add_argument("--prediction", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def load_json(path: Path, label: str) -> object:
    raw = path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raise SystemExit("[FAIL] {} 含 UTF-8 BOM，平台要求无 BOM：{}".format(label, path))
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise SystemExit("[FAIL] {} 不是合法 UTF-8：{}".format(label, e))
    except json.JSONDecodeError as e:
        raise SystemExit("[FAIL] {} 不是合法 JSON：{}".format(label, e))


def main() -> int:
    args = parse_args()
    pred_path, task_path = Path(args.prediction), Path(args.task)
    for p in (pred_path, task_path):
        if not p.exists():
            print("[FAIL] 文件不存在: {}".format(p))
            return 1

    pred = load_json(pred_path, "prediction")
    task = load_json(task_path, "task")

    if not isinstance(pred, dict):
        print("[FAIL] prediction 顶层必须是 JSON 对象，实际为 {}".format(type(pred).__name__))
        return 1
    if not isinstance(task, dict) or "queries" not in task or "gallery" not in task:
        print("[FAIL] task.json 缺少 queries / gallery")
        return 1

    queries, gallery = task["queries"], task["gallery"]
    q_ids = [q["query_id"] for q in queries]
    q_type = {q["query_id"]: q["query_type"] for q in queries}
    g_ids = {g["image_id"] for g in gallery}
    g_mod = {g["image_id"]: g["modality"] for g in gallery}

    # query 自身图像（工程泄漏检查，与推理脚本同一套解析口径）
    base = task_path.resolve().parent
    q_self = {q["query_id"]: (base / q["image_path"]).resolve() for q in queries}
    g_abs = {g["image_id"]: (base / g["image_path"]).resolve() for g in gallery}

    errors = []

    # 1. query 集合
    got, want = set(pred), set(q_ids)
    miss, extra = sorted(want - got), sorted(got - want)
    if miss:
        errors.append("缺少 {} 条 query（示例: {}）".format(len(miss), ", ".join(miss[:3])))
    if extra:
        errors.append("多出 {} 条非 task query（示例: {}）".format(len(extra), ", ".join(extra[:3])))

    # 2~6. 逐条候选
    MODE = {"not_list": 0, "bad_len": 0, "not_str": 0, "dup": 0, "unknown": 0,
            "mod_bad": 0, "self_hit": 0}
    bad_examples = []
    for qid in q_ids:
        if qid not in pred:
            continue
        lst = pred[qid]
        if not isinstance(lst, list):
            MODE["not_list"] += 1
            bad_examples.append("{}: 值不是数组".format(qid))
            continue
        if len(lst) != args.topk:
            MODE["bad_len"] += 1
            bad_examples.append("{}: 长度 {} != {}".format(qid, len(lst), args.topk))
        if any(not isinstance(i, str) for i in lst):
            MODE["not_str"] += 1
            bad_examples.append("{}: 存在非字符串候选（如相似度/路径等额外字段）".format(qid))
            continue
        if len(set(lst)) != len(lst):
            MODE["dup"] += 1
            bad_examples.append("{}: 候选重复".format(qid))
        unknown = [i for i in lst if i not in g_ids]
        if unknown:
            MODE["unknown"] += 1
            bad_examples.append("{}: 非 gallery id {}".format(qid, ", ".join(unknown[:2])))
        want_mod = TARGET_MOD.get(q_type.get(qid, ""))
        mod_bad = [i for i in lst if i in g_mod and g_mod[i] != want_mod]
        if mod_bad:
            MODE["mod_bad"] += 1
            bad_examples.append("{}: 模态错配 {} 个（应为 {}）".format(qid, len(mod_bad), want_mod))
        if len(set(g_abs[i] for i in lst if i in g_abs)) != len(lst):
            pass  # gallery 内部同图不同 id 属数据问题，不判无效
        if q_self.get(qid) in {g_abs[i] for i in lst if i in g_abs}:
            MODE["self_hit"] += 1
            bad_examples.append("{}: 候选中出现 query 自身图像".format(qid))

    if MODE["not_list"]:
        errors.append("{} 条 query 的值不是数组".format(MODE["not_list"]))
    if MODE["bad_len"]:
        errors.append("{} 条 query 候选数 != {}".format(MODE["bad_len"], args.topk))
    if MODE["not_str"]:
        errors.append("{} 条 query 含非字符串候选".format(MODE["not_str"]))
    if MODE["dup"]:
        errors.append("{} 条 query 候选重复".format(MODE["dup"]))
    if MODE["unknown"]:
        errors.append("{} 条 query 含非 gallery id".format(MODE["unknown"]))
    if MODE["mod_bad"]:
        errors.append("{} 条 query 候选模态错配".format(MODE["mod_bad"]))
    if MODE["self_hit"]:
        errors.append("{} 条 query 候选中含自身图像（泄漏）".format(MODE["self_hit"]))

    # ---- 报告 ----
    print("=" * 72)
    print("提交文件校验: {}".format(pred_path))
    print("=" * 72)
    print("  task.json        : {}".format(task_path))
    print("  query / gallery  : {} / {}".format(len(q_ids), len(g_ids)))
    print("  prediction query : {}".format(len(pred)))
    print("  候选长度         : 恰好 {}".format(args.topk))
    dist = Counter(q_type[q] for q in q_ids)
    print("  方向分布         : " + "  ".join(
        "{}={}".format(d, dist.get(d, 0)) for d in ("O2S", "S2O", "O2O")))
    print("  唯一候选总数     : {}".format(len({i for v in pred.values()
                                                if isinstance(v, list)
                                                for i in v if isinstance(i, str)})))
    print("-" * 72)

    if errors:
        print("结果: FAIL（{} 项）".format(len(errors)))
        for e in errors:
            print("  - {}".format(e))
        for b in bad_examples[:5]:
            print("    例: {}".format(b))
        return 1
    print("结果: PASS —— 满足官方全部「直接判定无效」条件，模态一致且无自身匹配")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())