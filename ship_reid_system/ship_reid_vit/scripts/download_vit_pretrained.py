# -*- coding: utf-8 -*-
"""下载并导出 ViT 预训练权重到本地 weights/ 目录（离线训练用）。

背景（技术方案 Step6：ViT 预训练权重）：
    双分支 ViT 需要 ImageNet 预训练权重作为初始化。为支持无网络/内网环境，
    本脚本把 timm 的在线预训练权重下载一次后导出为本地 .pth 文件，供
    config 中 `pretrained_path` 离线加载（见 models/backbone.py:build_backbone）。

用法：
    python scripts/download_vit_pretrained.py
    python scripts/download_vit_pretrained.py --name vit_base_patch16_224 --out weights/vit_base_patch16_224.pth
    python scripts/download_vit_pretrained.py --check-only   # 仅校验本地文件是否可用

产出：
    weights/vit_base_patch16_224.pth   内容 {"state_dict": {...}, "arch": "vit_base_patch16_224"}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_NAME = "vit_base_patch16_224"
DEFAULT_OUT = "weights/vit_base_patch16_224.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download & export ViT pretrained weights")
    p.add_argument("--name", type=str, default=DEFAULT_NAME, help="timm 模型名")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="本地导出路径（相对工程根目录）")
    p.add_argument("--check-only", action="store_true", help="仅校验本地权重，不下载")
    return p.parse_args()


def _build_base(name: str, pretrained: bool):
    import timm

    return timm.create_model(name, pretrained=pretrained)


def _self_check(model, name: str) -> None:
    """一次前向自检，确认权重可正常构建与推理。"""
    model.eval()
    size = 224
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        y = model(x)
    print(f"[自检] {name} 前向输出 shape = {tuple(y.shape)}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if args.check_only:
        if not out_path.exists():
            raise SystemExit(f"[check-only] 本地权重不存在：{out_path}")
        state = torch.load(out_path, map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        print(f"[check-only] 权重可用：{out_path}，共 {len(sd)} 个张量")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] 从 timm 拉取预训练权重：{args.name}（首次需联网）")
    model = _build_base(args.name, pretrained=True)
    _self_check(model, args.name)

    print(f"[2/3] 导出 state_dict -> {out_path}")
    torch.save({"state_dict": model.state_dict(), "arch": args.name}, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[3/3] 完成：{out_path}（{size_mb:.1f} MB）")


if __name__ == "__main__":
    main()