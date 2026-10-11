"""对比不同 epoch 的 CycleGAN checkpoint 翻译效果。

在 AutoDL 上运行：
    python3 compare_ckpt.py

输出：/root/autodl-tmp/cyclegan/checkpoints/ship_os/compare_*.png
每行：原图 | ep10 | ep20 | ep30
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import transforms

from models import ResnetGenerator

CKPT_DIR = Path("/root/autodl-tmp/cyclegan/checkpoints/ship_os")
DATA = Path("/root/autodl-tmp/cyclegan_data")
EPOCHS = [10, 20, 30]
N_SAMPLES = 6
SIZE = 256

tfm = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_g(name, device):
    g = ResnetGenerator(3, 3, 64, 9, nn.InstanceNorm2d)
    g.load_state_dict(torch.load(str(CKPT_DIR / name), map_location=device))
    g.to(device).eval()
    return g


def translate(g, img, device):
    t = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = g(t)
    out = out.squeeze(0).cpu().float().clamp(-1, 1)
    arr = (out.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
    return Image.fromarray(arr.astype(np.uint8))


def pick_samples(domain):
    files = sorted((DATA / domain).glob("*.tif"))
    step = len(files) // N_SAMPLES
    return [files[i * step] for i in range(N_SAMPLES)]


def make_grid(domain, ckpt_letter, out_name, device):
    # train.py 实际约定：G_A: A->B (optical->sar)，G_B: B->A (sar->optical)
    files = pick_samples(domain)
    gens = {}
    for ep in EPOCHS:
        ckpt = f"netG_{ckpt_letter}_ep{ep}.pth"
        if (CKPT_DIR / ckpt).exists():
            gens[ep] = load_g(ckpt, device)
        else:
            print(f"缺失: {ckpt}")

    cols = 1 + len(gens)
    header_h, pad = 28, 4
    grid = Image.new("RGB", (cols * SIZE + (cols + 1) * pad,
                             N_SAMPLES * SIZE + header_h + (N_SAMPLES + 1) * pad), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((pad + SIZE // 2 - 25, 6), "original", fill="black")
    for i, ep in enumerate(gens):
        x = pad + (i + 1) * (SIZE + pad) + SIZE // 2 - 12
        draw.text((x, 6), f"ep{ep}", fill="black")

    for r, f in enumerate(files):
        y = header_h + pad + r * (SIZE + pad)
        img = Image.open(f).convert("RGB").resize((SIZE, SIZE))
        grid.paste(img, (pad, y))
        for c, ep in enumerate(gens):
            x = pad + (c + 1) * (SIZE + pad)
            fake = translate(gens[ep], img, device)
            grid.paste(fake, (x, y))

    out_path = CKPT_DIR / out_name
    grid.save(out_path)
    print(f"saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    # optical -> fake sar: G_A (A->B)
    make_grid("trainA", "A", "compare_optical_to_sar.png", device)
    # sar -> fake optical: G_B (B->A)
    make_grid("trainB", "B", "compare_sar_to_optical.png", device)
    print("完成")


if __name__ == "__main__":
    main()
