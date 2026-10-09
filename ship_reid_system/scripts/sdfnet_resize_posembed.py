# -*- coding: utf-8 -*-
"""离线把 SDF-Net 官方 checkpoint 的 pos_embed 插值到目标输入尺寸。

背景（为什么需要这个脚本）：
  - SDF-Net 的 pos_embed 形状是 [1, num_patches + 2, 768]，前缀 2 个 token 为
    cls_token 与 disentangle 的 spe_token；wh_token 是在 pos_embed 相加之后才
    拼接的，所以不占 pos_embed。
  - 官方权重在 256x128 / stride 16 下训练：num_y=16, num_x=8 -> num_patches=128
    -> pos_embed = [1, 130, 768]。
  - 改成 256x256 后 num_patches=256 -> 需要 [1, 258, 768]，必须插值。
  - SDF-Net 自带的 resize_pos_embed 假设「前缀只有 1 个 token」且「网格为正方形」，
    对本例（前缀 2 个 token、16x8 非方阵）会直接报错。
  - 因此在离线把 pos_embed 转换好后直接保存成新权重，训练/推理继续走 'clip'
    严格加载路径（Backbone.load_param），不需要改动模型代码。

坐标约定：reshape 用 (num_y, num_x) = (h/patch, w/patch) 的行优先顺序，
与 PatchEmbed 的 flatten(2) 一致（num_y 为行数、num_x 为列数）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def resize_pos_embed_disentangle(posemb: torch.Tensor, num_prefix: int,
                                 src_hw: tuple[int, int],
                                 tgt_hw: tuple[int, int]) -> torch.Tensor:
    """把 posemb 的 patch 网格从 src_hw 双线性插值到 tgt_hw，前缀 token 原样保留。"""
    n_src = src_hw[0] * src_hw[1]
    if posemb.dim() != 3:
        raise ValueError(f"pos_embed 期望 3 维，实际 {tuple(posemb.shape)}")
    if posemb.shape[1] != num_prefix + n_src:
        raise ValueError(
            f"pos_embed token 数不匹配：{posemb.shape[1]} != {num_prefix} + {n_src} "
            f"(src_hw={src_hw})"
        )
    prefix = posemb[:, :num_prefix]
    grid = posemb[:, num_prefix:]
    c = grid.shape[-1]
    grid = grid.reshape(1, src_hw[0], src_hw[1], c).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=tgt_hw, mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, tgt_hw[0] * tgt_hw[1], c)
    return torch.cat([prefix, grid], dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDF-Net pos_embed 尺寸转换（256x128 -> 256x256）")
    p.add_argument("--in_weight", required=True, help="输入权重（官方 SDF-Net.pth）")
    p.add_argument("--out_weight", required=True, help="输出权重路径")
    p.add_argument("--src_size", type=int, nargs=2, default=[256, 128],
                   help="原始训练输入尺寸 H W（默认 256 128）")
    p.add_argument("--tgt_size", type=int, nargs=2, default=[256, 256],
                   help="目标训练输入尺寸 H W（默认 256 256）")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--num_prefix", type=int, default=2,
                   help="pos_embed 前缀 token 数：disentangle=True 时为 2（cls + spe）")
    p.add_argument("--key", type=str, default="base.pos_embed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_hw = (args.src_size[0] // args.patch_size, args.src_size[1] // args.patch_size)
    tgt_hw = (args.tgt_size[0] // args.patch_size, args.tgt_size[1] // args.patch_size)

    in_path = Path(args.in_weight)
    out_path = Path(args.out_weight)
    sd = torch.load(str(in_path), map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if args.key not in sd:
        raise KeyError(f"权重中找不到 {args.key}，现有 pos_embed 相关键："
                       f"{[k for k in sd if 'pos_embed' in k]}")

    old = sd[args.key]
    new = resize_pos_embed_disentangle(old, args.num_prefix, src_hw, tgt_hw)
    sd[args.key] = new

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, str(out_path))

    print(f"pos_embed: {tuple(old.shape)} {src_hw} -> {tuple(new.shape)} {tgt_hw}")
    print(f"已保存: {out_path}")
    print(f"键数: {len(sd)}（仅 pos_embed 被替换）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())