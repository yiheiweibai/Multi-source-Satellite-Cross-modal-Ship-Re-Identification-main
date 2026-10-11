"""赛题提交评分（独立版，无 torch 依赖）。

复刻 utils/metrics.py::evaluate_submission：
  方向 O2S/S2O/O2O 各算 R@1 与 mAP@10，方向得分=两者均值，
  综合 = 0.45*O2S + 0.45*S2O + 0.10*O2O。
用法: python score_submission.py --prediction pred.json --task task.json --gt gt.json
"""
import argparse
import json


def map_at_k(hits, num_rel, k=10):
    num_rel = int(num_rel)
    if num_rel <= 0:
        return 0.0
    k = min(k, len(hits))
    if k <= 0:
        return 0.0
    hit, ap = 0, 0.0
    for pos in range(k):
        if hits[pos]:
            hit += 1
            ap += hit / (pos + 1)
    return ap / min(num_rel, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    pred = json.load(open(args.prediction, encoding="utf-8"))
    task = json.load(open(args.task, encoding="utf-8"))
    gt = {k: set(v) for k, v in json.load(open(args.gt, encoding="utf-8")).items()}
    queries = task["queries"]

    weights = {"O2S": 0.45, "S2O": 0.45, "O2O": 0.10}
    result = {}
    for d, w in weights.items():
        sub = [q for q in queries if q.get("query_type") == d]
        n = max(len(sub), 1)
        r1, ap_sum = 0, 0.0
        for q in sub:
            cands = pred.get(q["query_id"], [])[:10]
            pos = gt.get(q["query_id"], set())
            hits = [1 if c in pos else 0 for c in cands]
            if hits and hits[0] == 1:
                r1 += 1
            ap_sum += map_at_k(hits, len(pos), 10)
        r1 /= n
        mAP = ap_sum / n
        result[d] = (len(sub), r1, mAP, (r1 + mAP) / 2, w)

    print("=" * 60)
    print(f"{'方向':<6}{'Query数':<8}{'R@1':<10}{'mAP@10':<12}{'方向得分':<10}{'权重':<6}")
    overall = 0.0
    for d in ("O2S", "S2O", "O2O"):
        cnt, r1, mAP, sc, w = result[d]
        overall += sc * w
        print(f"{d:<6}{cnt:<8}{r1:<10.4f}{mAP:<12.4f}{sc:<10.4f}{w:<6.2f}")
    print("-" * 60)
    print(f"综合得分: {overall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
