"""下载 SDF-Net 官方预训练权重（HuggingFace: Chenfree233/SDF-Net）。

用法：
    python scripts/sdfnet_download_weights.py [--out <路径>]

默认输出：ship_reid_system/SDF-Net/logs/SDF-Net/SDF-Net.pth
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF_URL = "https://huggingface.co/Chenfree233/SDF-Net/resolve/main/SDF-Net.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载 SDF-Net 官方预训练权重")
    default = Path(__file__).resolve().parent.parent / "SDF-Net" / "logs" / "SDF-Net" / "SDF-Net.pth"
    p.add_argument("--out", type=str, default=str(default))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if out.exists() and out.stat().st_size > 100_000_000:
        print(f"权重已存在，跳过下载: {out} ({out.stat().st_size / 1048576:.1f} MB)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {HF_URL}")
    print(f"保存到 {out}")
    req = urllib.request.Request(HF_URL, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r{done / 1048576:.1f} / {total / 1048576:.1f} MB ({pct}%)", end="", flush=True)
    print()
    tmp.replace(out)
    print(f"下载完成: {out} ({out.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
