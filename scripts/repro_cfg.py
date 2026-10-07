# -*- coding: utf-8 -*-
"""configs/reproduce.yaml 的读取、导出与执行工具（14~18 号 .bat 的后端）。

把 yaml 里的路径与融合配方解析成可直接消费的形式，避免两件事：
  1) 路径被硬编码进批处理脚本（项目硬约束：路径统一放 yaml）；
  2) --members / --weights 在命令行上被手写/展开错位
     （cmd 会对内嵌引号做二次解析，列表类参数不放命令行，直接在 Python 侧消费）。

集成配方结构：ensemble.members 是成员池（name/ckpt/pred，定义一次），
ensemble.presets 是「预设名 -> {成员名: 权重}」，ensemble.active 选择生效预设。
成员与权重按预设内的书写顺序绑定，由本脚本解析，不经过 shell。

子命令：
    dump  [--env-out PATH]  写出 set 变量的 .bat 与同名 _steps.txt（逐 checkpoint 推理步骤）
    check                校验当前预设所需的输入是否齐备（缺任一则退出码 1）
    fuse                 按当前预设执行 RRF 融合并写出最终 prediction.json
    show                 打印解析后的完整配置（含成员池与全部预设）

通用参数：
    --preset NAME        覆盖 ensemble.active
    --weights W1 W2 ...  覆盖当前预设的权重（个数须与预设成员数一致，按序对应）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

SHIP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CFG = SHIP_DIR / "configs" / "reproduce.yaml"


def load(cfg_path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML 未安装，请执行: python -m pip install pyyaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(value: str) -> Path:
    """相对路径一律相对工程根目录解析。"""
    p = Path(value)
    return p.resolve() if p.is_absolute() else (SHIP_DIR / p).resolve()


def disp_len(s: str) -> int:
    """终端显示宽度（CJK 全角字符占 2 列），用于对齐输出。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def member_pool(cfg: dict) -> dict:
    pool = {}
    for m in cfg["ensemble"]["members"]:
        name = m["name"]
        if name in pool:
            raise SystemExit("成员池存在重名成员: {}".format(name))
        pool[name] = m
    return pool


def resolve_members(cfg: dict, preset: str = "", weights: list | None = None) -> tuple[str, list]:
    """按 active 预设（或 --preset 覆盖）解析参与融合的成员。

    返回 (预设名, members)；members 元素为 (成员名, 预测绝对路径, 权重)。
    """
    ens = cfg["ensemble"]
    pool = member_pool(cfg)
    presets = ens.get("presets") or {}

    name = preset or str(ens.get("active") or "")
    if not name:
        raise SystemExit("ensemble.active 未设置，且未指定 --preset")
    if name not in presets:
        raise SystemExit("预设不存在: {}（可用: {}）".format(
            name, ", ".join(presets) or "无"))
    spec = presets[name] or {}
    if not spec:
        raise SystemExit("预设 {} 为空，至少需要一个成员".format(name))

    out = []
    for mname, w in spec.items():
        if mname not in pool:
            raise SystemExit("预设 {} 引用了未定义成员: {}（成员池: {}）".format(
                name, mname, ", ".join(pool)))
        w = float(w)
        if w <= 0:
            raise SystemExit("预设 {} 的成员 {} 权重必须 > 0（当前 {}）".format(name, mname, w))
        out.append((mname, str(resolve(pool[mname]["pred"])), w))

    if weights:
        if len(weights) != len(out):
            raise SystemExit("--weights 需要 {} 个值（当前 {} 个），顺序对应: {}".format(
                len(out), len(weights), ", ".join(n for n, _, _ in out)))
        out = [(n, p, float(w)) for (n, p, _), w in zip(out, weights)]
        if any(w <= 0 for _, _, w in out):
            raise SystemExit("--weights 中的权重必须 > 0")
    return name, out


def build(cfg: dict, preset: str = "", weights: list | None = None):
    """返回 (变量字典, 推理步骤, 融合成员, 预设名)。

    steps   元素为 (成员名, ckpt绝对路径, 预测输出绝对路径)   —— 必须本地推理的成员
    members 元素为 (成员名, 预测绝对路径, 权重)              —— 当前预设的全部成员
    """
    p, ens = cfg["paths"], cfg["ensemble"]
    sdfnet_dir = resolve(p["sdfnet_dir"])
    shipvit_dir = resolve(p["shipvit_dir"])
    ckpt_dir = resolve(p["sdf_checkpoint_dir"])
    pool = member_pool(cfg)
    preset_name, members = resolve_members(cfg, preset, weights)

    v = {
        "REPRO_DIR": str(SHIP_DIR),
        "SDFNET_DIR": str(sdfnet_dir),
        "SDFNET_CONFIG": p["sdfnet_config"],
        "SDFNET_OFFICIAL": str(resolve(p["sdfnet_official"])),
        "SDF_CKPT_DIR": str(ckpt_dir),
        "SDF_SIM_DIR": str(resolve(p["sdf_sim_dir"])),
        "TEST_TASK": str(resolve(p["test_task"])),
        "SHIPVIT_DIR": str(shipvit_dir),
        "SHIPVIT_PY": str(shipvit_dir / "venv" / "Scripts" / "python.exe"),
        "SHIPVIT_CKPT": str(shipvit_dir / "outputs" / "checkpoints" / "best.pth"),
        "SHIPVIT_PRED": str(resolve(p["shipvit_pred"])),
        "OUT_PRED": str(resolve(p["out_prediction"])),
        "RRF_K": str(ens["k"]),
        "RRF_TOP_K": str(ens["topk"]),
        "RRF_PRESET": preset_name,
        "LOCAL_VAL_TASK": str(resolve(cfg.get("local_val", {}).get("task", ""))),
        "LOCAL_VAL_FINAL": str(cfg.get("local_val", {}).get("final", "")),
        "LOCAL_VAL_FOLD0": str(cfg.get("local_val", {}).get("fold0", "")),
        "LOCAL_VAL_FOLD1": str(cfg.get("local_val", {}).get("fold1", "")),
    }

    # 当前预设是否用到 ship_reid_vit 分支（按预测路径判定，不依赖成员名）
    v["SHIPVIT_NEEDED"] = "1" if any(p == v["SHIPVIT_PRED"] for _, p, _ in members) else "0"

    steps = []
    for mname, pred, _ in members:
        ck = pool[mname].get("ckpt")
        if ck:
            steps.append((mname, str(ckpt_dir / ck), pred))

    v["SDF_N"] = str(len(steps))
    v["RRF_N"] = str(len(members))
    v["RRF_MEMBERS"] = " ".join("{}({})".format(n, w) for n, _, w in members)
    v["RRF_WEIGHTS"] = " ".join(str(w) for _, _, w in members)
    return v, steps, members, preset_name


def required_items(v: dict, steps: list, members: list) -> list:
    """(说明, 路径, 是否必须)。

    有 ckpt 的成员由 15 号脚本本地推理生成；无 ckpt 的成员（如 ship_reid_vit）
    其预测必须已存在，或由 16 号脚本补出。
    """
    generated = {n for n, _, _ in steps}
    shipvit_active = v.get("SHIPVIT_NEEDED", "1") == "1"

    items = [
        ("SDF-Net venv python", Path(v["REPRO_DIR"]) / ".venv-sdfnet" / "Scripts" / "python.exe", True),
        ("SDF-Net config", Path(v["SDFNET_DIR"]) / v["SDFNET_CONFIG"], True),
        ("SDF-Net official weight", Path(v["SDFNET_OFFICIAL"]), True),
        ("competition test task.json", Path(v["TEST_TASK"]), True),
    ]
    if shipvit_active:
        items += [
            ("ship_reid_vit venv python", Path(v["SHIPVIT_PY"]), True),
            ("ship_reid_vit best.pth", Path(v["SHIPVIT_CKPT"]), True),
        ]
    items += [("SDF-Net ckpt {}".format(n), Path(c), True) for n, c, _ in steps]
    items += [("member pred {}".format(n), Path(p), n not in generated)
              for n, p, _ in members]
    items += [("final submission (auto by step 17)", Path(v["OUT_PRED"]), False)]
    return items


def do_dump(v: dict, steps: list, out: Path) -> int:
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="ascii", newline="\r\n") as f:
            f.write("@echo off\r\n")
            f.write("REM generated by scripts/repro_cfg.py -- do not edit\r\n")
            for k, val in v.items():
                f.write('set "{}={}"\r\n'.format(k, val))
        steps_path = out.with_name(out.stem + "_steps.txt")
        with open(steps_path, "w", encoding="ascii", newline="\n") as f:
            for name, ckpt, pred in steps:
                f.write("{}|{}|{}\n".format(name, ckpt, pred))
    except UnicodeEncodeError as e:
        print("[repro_cfg] 路径含非 ASCII 字符，无法写出 cmd 脚本: {}".format(e))
        return 1
    print("[repro_cfg] preset: {}".format(v["RRF_PRESET"]))
    print("[repro_cfg] env   -> {}".format(out))
    print("[repro_cfg] steps -> {}".format(steps_path))
    return 0


def do_check(v: dict, steps: list, members: list) -> int:
    items = required_items(v, steps, members)
    width = max(disp_len(desc) for desc, _, _ in items)
    missing = []
    print("active preset: {}  ({} members)".format(v["RRF_PRESET"], v["RRF_N"]))
    print("-" * 78)
    for desc, path, must in items:
        ok = path.exists()
        if not ok and must:
            missing.append(desc)
        flag = "OK  " if ok else ("MISS" if must else "----")
        pad = " " * (width - disp_len(desc))
        print("  [{}] {}{}  {}".format(flag, desc, pad, path))
    print("-" * 78)
    if missing:
        print("缺失必需输入 {} 项: {}".format(len(missing), ", ".join(missing)))
        return 1
    print("全部必需输入齐备。")
    return 0


def do_fuse(v: dict, members: list, out_path: str = "") -> int:
    """按当前预设执行 RRF 融合（成员与权重按序绑定，不经过 shell）。"""
    spec = importlib.util.spec_from_file_location(
        "rrf_fuse", str(Path(__file__).resolve().parent / "rrf_fuse.py"))
    rrf_fuse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rrf_fuse)

    for name, path, _ in members:
        if not Path(path).exists():
            print("[repro_cfg] 成员预测不存在: {} ({})".format(path, name))
            return 1

    lists = [rrf_fuse.load_pred(p) for _, p, _ in members]
    weights = [w for _, _, w in members]
    k, topk = float(v["RRF_K"]), int(v["RRF_TOP_K"])

    base = set(lists[0])
    for (name, path, _), lst in zip(members[1:], lists[1:]):
        if set(lst) != base:
            print("[repro_cfg] query 集合不一致: {}".format(path))
            return 1

    pred = rrf_fuse.rrf(lists, weights, k, topk)
    print("preset: {}  ({} members, k={})".format(v["RRF_PRESET"], len(members), k))
    for name, path, w in members:
        print("  member w={:<5} {:<16} {}".format(w, name, path))
    print("RRF fusion done: {} queries".format(len(pred)))

    rrf_fuse.verify(pred, v["TEST_TASK"], topk)

    out = Path(out_path) if out_path else Path(v["OUT_PRED"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)
    print("written: {}".format(out))
    return 0


def do_show(v: dict, steps: list, cfg: dict) -> int:
    width = max(disp_len(k) for k in v)
    for k in sorted(v):
        print("{}{} = {}".format(k, " " * (width - disp_len(k)), v[k]))

    pool = member_pool(cfg)
    print("\nmember pool ({}):".format(len(pool)))
    for name, m in pool.items():
        print("  {:<16} ckpt={:<24} pred={}".format(
            name, m.get("ckpt") or "-", m["pred"]))

    print("\npresets:")
    for pname, spec in (cfg["ensemble"].get("presets") or {}).items():
        mark = " *active" if pname == v["RRF_PRESET"] else ""
        body = ", ".join("{}={}".format(k, w) for k, w in (spec or {}).items())
        print("  {:<16}{}{}".format(pname, body, mark))

    print("\nSDF-Net inference steps for the active preset ({}):".format(len(steps)))
    for name, ckpt, pred in steps:
        print("  {} -> {}".format(name, Path(pred).name))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="reproduce.yaml reader / dumper / executor")
    ap.add_argument("cmd", choices=["dump", "check", "fuse", "show"])
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--preset", default="", help="覆盖 ensemble.active")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="覆盖当前预设的权重（个数须与预设成员数一致）")
    ap.add_argument("--env-out", dest="env_out", default="",
                    help="dump 的输出路径（env 脚本）；默认 %TEMP%\\shipreid_repro_env.bat")
    ap.add_argument("--out", default="",
                    help="fuse 的输出路径（预测 json）；默认用 ensemble 的 out_prediction")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print("[repro_cfg] 配置不存在: {}".format(cfg_path))
        return 1

    cfg = load(cfg_path)
    v, steps, members, _ = build(cfg, args.preset, args.weights)

    if args.cmd == "dump":
        if args.env_out:
            out = Path(args.env_out)
        else:
            import os
            tmp = os.environ.get("TEMP") or os.environ.get("TMP") or "."
            out = Path(tmp) / "shipreid_repro_env.bat"
        return do_dump(v, steps, out)
    if args.cmd == "check":
        return do_check(v, steps, members)
    if args.cmd == "fuse":
        return do_fuse(v, members, args.out)
    return do_show(v, steps, cfg)


if __name__ == "__main__":
    sys.exit(main())