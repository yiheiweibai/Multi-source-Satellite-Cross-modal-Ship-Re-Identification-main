"""TransOSS 一键准备脚本：在任意 Windows 机器（含另一台 4070 Ti SUPER）上准备开源模型运行环境。

完成：
  1. 克隆官方 Hoss-ReID 仓库（若缺失）
  2. 幂等安装 TransOSS 额外依赖（einops/yacs）
  3. 将赛题训练数据转换为 TransOSS 格式（transoss_data/HOSS/bounding_box_train，若缺失）
  4. 创建评估占位目录 query / bounding_box_test（从 bounding_box_train 复制 300+300 张，
     防止 HOSS 训练在 EVAL_PERIOD 时因空目录崩溃；占位数据不影响训练质量）
  5. 生成 Hoss-ReID/configs/hoss_transoss_competition.yml（ROOT_DIR 自动替换为绝对路径，
     避免跨目录相对路径解析失败）
  6. 复制 transoss_inference.py 到 Hoss-ReID 根目录
  7. 检查预训练权重 vit_b512_pre.pth，缺失时给出下载提示

用法（在 ship_reid_system 目录，先运行 01_setup.bat 建好 .venv）：
    .venv\\Scripts\\python.exe scripts/transoss_setup.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
SYS = SELF.parent.parent          # ship_reid_system
PROJ = SYS.parent                 # 项目根（赛题6-初赛 所在）
HOSS = SYS / "Hoss-ReID"          # 官方仓库克隆位置（与 07_transoss_pipeline.bat 注释一致）
DATA_ROOT = PROJ / "transoss_data" / "HOSS"
TRAIN_DIR = DATA_ROOT / "bounding_box_train"
QUERY_DIR = DATA_ROOT / "query"
TEST_DIR = DATA_ROOT / "bounding_box_test"
WEIGHT = HOSS / "weights" / "vit_b512_pre.pth"
CFG_TEMPLATE = SYS / "transoss_config" / "hoss_transoss_competition.yml"
CFG_DST = HOSS / "configs" / "hoss_transoss_competition.yml"
INFERENCE_SRC = SYS / "scripts" / "transoss_inference.py"
INFERENCE_DST = HOSS / "transoss_inference.py"
LABELS_TRAIN = PROJ / "赛题6-初赛" / "训练数据" / "labels_train.csv"
PREPARE_SCRIPT = SYS / "scripts" / "transoss_prepare_data.py"

PLACEHOLDER_N = 300


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    print(f"  $ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=cwd)
        return r.returncode == 0
    except FileNotFoundError as e:
        print(f"  执行失败（找不到命令）: {e}")
        return False


def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")


def main() -> int:
    print("==== TransOSS 一键准备 ====")
    print(f"项目根        : {PROJ}")
    print(f"Hoss-ReID 目标: {HOSS}")

    # 0. 前置检查
    if not LABELS_TRAIN.exists():
        print(f"\n[错误] 未找到赛题训练标注: {LABELS_TRAIN}")
        print("       请先同步项目仓库并放置赛题数据目录（赛题6-初赛/训练数据/labels_train.csv）。")
        return 1

    # 1. 克隆 Hoss-ReID
    step(1, 7, "克隆 Hoss-ReID 仓库")
    if HOSS.exists() and (HOSS / ".git").exists():
        print(f"  已存在，跳过克隆: {HOSS}")
    else:
        if not run(["git", "clone", "https://github.com/Alioth2000/Hoss-ReID.git", str(HOSS)]):
            print("  克隆失败：请检查网络/代理（若本机配置了 http.proxy 且代理未运行，用 git -c http.proxy= 直连）。")
            return 1

    # 2. 安装 TransOSS 额外依赖
    step(2, 7, "安装 TransOSS 依赖（einops / yacs，幂等）")
    py = str(SYS / ".venv" / "Scripts" / "python.exe")
    if not Path(py).exists():
        print(f"  [错误] 未找到虚拟环境: {py}")
        print("         请先运行 01_setup.bat 创建 .venv 并安装基础依赖。")
        return 1
    if not run([py, "-m", "pip", "install", "einops>=0.7.0", "yacs>=0.1.8"]):
        print("  依赖安装失败，请检查网络后重试。")
        return 1

    # 3. 数据转换
    step(3, 7, "转换训练数据为 TransOSS 格式")
    if TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*.tif")):
        print(f"  已存在，跳过转换: {TRAIN_DIR}")
    else:
        if not run([py, str(PREPARE_SCRIPT), "--csv", str(LABELS_TRAIN), "--out_dir", str(TRAIN_DIR)]):
            print("  数据转换失败。")
            return 1

    # 4. 评估占位目录（防 EVAL_PERIOD 崩溃）
    step(4, 7, "创建评估占位目录 query / bounding_box_test")
    for name, dst, skip in (("query", QUERY_DIR, PLACEHOLDER_N),
                            ("bounding_box_test", TEST_DIR, 2 * PLACEHOLDER_N)):
        if dst.exists() and any(dst.glob("*.tif")):
            print(f"  {name} 已存在，跳过: {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(TRAIN_DIR.glob("*.tif"))
        picked = files[:skip][:PLACEHOLDER_N] if name == "query" else files[PLACEHOLDER_N:skip]
        for f in picked:
            shutil.copy2(f, dst / f.name)
        print(f"  {name}: 已复制 {len(picked)} 张占位图（仅用于训练内评估，数字不可信）")

    # 5. 生成训练配置（ROOT_DIR 绝对路径）
    step(5, 7, "生成 Hoss-ReID 训练配置")
    if not CFG_TEMPLATE.exists():
        print(f"  [错误] 找不到配置模板: {CFG_TEMPLATE}")
        return 1
    text = CFG_TEMPLATE.read_text(encoding="utf-8")
    root_abs = DATA_ROOT.parent.as_posix()
    old = "ROOT_DIR: ('../transoss_data')"
    new = f"ROOT_DIR: ('{root_abs}')"
    if old not in text:
        print("  [警告] 模板中未找到预期 ROOT_DIR 行，跳过替换（请人工核对配置）。")
    else:
        text = text.replace(old, new)
    (HOSS / "configs").mkdir(parents=True, exist_ok=True)
    CFG_DST.write_text(text, encoding="utf-8")
    print(f"  ROOT_DIR -> {root_abs}")
    print(f"  已写入: {CFG_DST}")

    # 6. 复制推理脚本
    step(6, 7, "复制推理脚本到 Hoss-ReID 根目录")
    if not INFERENCE_SRC.exists():
        print(f"  [错误] 找不到推理脚本: {INFERENCE_SRC}")
        return 1
    shutil.copy2(INFERENCE_SRC, INFERENCE_DST)
    print(f"  已复制: {INFERENCE_DST}")

    # 7. 权重检查
    step(7, 7, "检查预训练权重")
    if WEIGHT.exists():
        print(f"  权重已存在: {WEIGHT}")
    else:
        print(f"  [提示] 未找到预训练权重: {WEIGHT}")
        print("         请从 TransOSS 官方渠道下载 vit_b512_pre.pth（GitHub: Alioth2000/Hoss-ReID README；官方数据与预训练权重托管在 Zenodo: https://zenodo.org/records/15860212），")
        print("         下载后放到该路径即可。没有权重时微调将退化为 ImageNet 随机初始化，精度显著下降。")

    print("\n==== 准备完成 ====")
    print("下一步（自动训练 + 推理）运行：")
    print("  07_transoss_pipeline.bat")
    print("或手动进入 Hoss-ReID 目录执行：")
    print(f"  cd /d \"{HOSS}\"")
    print("  ..\\.venv\\Scripts\\python.exe train.py --config_file configs/hoss_transoss_competition.yml")
    print("  ..\\.venv\\Scripts\\python.exe transoss_inference.py --config_file configs/hoss_transoss_competition.yml \\")
    print("      --weight logs/competition_transoss/transformer_200.pth --task_json \"../../赛题6-初赛/初赛测试数据/task.json\" \\")
    print("      --out_prediction ../prediction_transoss.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
