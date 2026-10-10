"""CycleGAN 训练脚本（AutoDL / 本地 GPU 通用）。

用法（AutoDL）：
    cd /root/autodl-tmp/cyclegan
    python train.py --dataroot /root/autodl-tmp/cyclegan_data \
        --name ship_os --n_epochs 100 --n_epochs_decay 100 --batch_size 4

用法（本地 4070）：
    python cyclegan/train.py --dataroot cyclegan_data --name ship_os --batch_size 4

输出：checkpoints/<name>/ 下保存 netG_A.pth, netG_B.pth, netD_A.pth, netD_B.pth
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import GANLoss, build_cyclegan
from dataset import UnpairedDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True, help="cyclegan_data 目录")
    p.add_argument("--name", type=str, default="ship_os", help="实验名")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=100, help="初始 lr 保持的 epoch 数")
    p.add_argument("--n_epochs_decay", type=int, default=100, help="lr 线性衰减的 epoch 数")
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--lambda_A", type=float, default=10.0, help="cycle loss A->B->A 权重")
    p.add_argument("--lambda_B", type=float, default=10.0, help="cycle loss B->A->B 权重")
    p.add_argument("--lambda_identity", type=float, default=0.5, help="identity loss 权重")
    p.add_argument("--pool_size", type=int, default=50, help="图像缓冲池大小")
    p.add_argument("--load_size", type=int, default=286)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--no_flip", action="store_true")
    p.add_argument("--save_epoch_freq", type=int, default=10)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--gpu_ids", type=str, default="0")
    return p.parse_args()


class ImagePool:
    """历史图像缓冲池，稳定判别器训练。"""

    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor):
        if self.pool_size == 0:
            return images
        ret = []
        for img in images:
            img = torch.unsqueeze(img.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                ret.append(img)
            else:
                p = torch.rand(1).item()
                if p > 0.5:
                    idx = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[idx].clone()
                    self.images[idx] = img
                    ret.append(tmp)
                else:
                    ret.append(img)
        return torch.cat(ret, dim=0)


def set_requires_grad(nets, requires_grad: bool):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CycleGAN] device={device}, dataroot={args.dataroot}")

    # 数据
    dataset = UnpairedDataset(args.dataroot, "train", args.load_size, args.crop_size, args.no_flip)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
    print(f"[CycleGAN] train: A={dataset.A_size}, B={dataset.B_size}, batches={len(loader)}")

    # 模型
    g_a, g_b, d_a, d_b = build_cyclegan()
    g_a, g_b, d_a, d_b = g_a.to(device), g_b.to(device), d_a.to(device), d_b.to(device)

    criterion_gan = GANLoss().to(device)
    criterion_cycle = torch.nn.L1Loss()
    criterion_idt = torch.nn.L1Loss()

    opt_g = torch.optim.Adam(itertools.chain(g_a.parameters(), g_b.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(d_a.parameters(), d_b.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))

    # LR 调度：线性衰减
    def lambda_rule(epoch):
        total = args.n_epochs + args.n_epochs_decay
        if epoch <= args.n_epochs:
            return 1.0
        return 1.0 - (epoch - args.n_epochs) / max(1, args.n_epochs_decay)

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda_rule)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda_rule)

    fake_a_pool = ImagePool(args.pool_size)
    fake_b_pool = ImagePool(args.pool_size)

    ckpt_dir = Path(args.checkpoints_dir) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = args.n_epochs + args.n_epochs_decay
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        losses = {"G": 0.0, "D": 0.0, "cycle": 0.0, "idt": 0.0}
        n = 0
        for i, data in enumerate(loader, 1):
            real_a = data["A"].to(device)
            real_b = data["B"].to(device)

            # ---- G_A 和 G_B ----
            set_requires_grad([d_a, d_b], False)
            opt_g.zero_grad()

            # identity loss
            if args.lambda_identity > 0:
                idt_a = g_a(real_b)
                idt_b = g_b(real_a)
                loss_idt_a = criterion_idt(idt_a, real_b) * args.lambda_B * args.lambda_identity
                loss_idt_b = criterion_idt(idt_b, real_a) * args.lambda_A * args.lambda_identity
            else:
                loss_idt_a = loss_idt_b = torch.tensor(0.0, device=device)

            # GAN loss
            fake_b = g_a(real_a)
            pred_fake = d_a(fake_b)
            loss_g_a = criterion_gan(pred_fake, True)
            fake_a = g_b(real_b)
            pred_fake = d_b(fake_a)
            loss_g_b = criterion_gan(pred_fake, True)

            # cycle loss
            rec_a = g_b(fake_b)
            loss_cycle_a = criterion_cycle(rec_a, real_a) * args.lambda_A
            rec_b = g_a(fake_a)
            loss_cycle_b = criterion_cycle(rec_b, real_b) * args.lambda_B

            loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
            loss_g.backward()
            opt_g.step()

            # ---- D_A 和 D_B ----
            set_requires_grad([d_a, d_b], True)
            opt_d.zero_grad()

            pred_real = d_a(real_b)
            loss_d_real = criterion_gan(pred_real, True)
            fake_b_ = fake_b_pool.query(fake_b.detach())
            pred_fake = d_a(fake_b_)
            loss_d_fake = criterion_gan(pred_fake, False)
            loss_d_a = (loss_d_real + loss_d_fake) * 0.5
            loss_d_a.backward()

            pred_real = d_b(real_a)
            loss_d_real = criterion_gan(pred_real, True)
            fake_a_ = fake_a_pool.query(fake_a.detach())
            pred_fake = d_b(fake_a_)
            loss_d_fake = criterion_gan(pred_fake, False)
            loss_d_b = (loss_d_real + loss_d_fake) * 0.5
            loss_d_b.backward()
            opt_d.step()

            losses["G"] += loss_g.item()
            losses["D"] += (loss_d_a.item() + loss_d_b.item())
            losses["cycle"] += (loss_cycle_a.item() + loss_cycle_b.item())
            losses["idt"] += (loss_idt_a.item() + loss_idt_b.item())
            n += 1

            if i % args.print_freq == 0:
                print(f"  ep{epoch} [{i}/{len(loader)}] "
                      f"G={losses['G']/n:.3f} D={losses['D']/n:.3f} "
                      f"cyc={losses['cycle']/n:.3f} idt={losses['idt']/n:.3f}")

        sched_g.step()
        sched_d.step()
        dt = time.time() - t0
        print(f"[Epoch {epoch}/{total_epochs}] "
              f"G={losses['G']/n:.3f} D={losses['D']/n:.3f} "
              f"cyc={losses['cycle']/n:.3f} idt={losses['idt']/n:.3f} "
              f"lr={opt_g.param_groups[0]['lr']:.2e} {dt:.1f}s")

        if epoch % args.save_epoch_freq == 0 or epoch == total_epochs:
            torch.save(g_a.state_dict(), ckpt_dir / f"netG_A_ep{epoch}.pth")
            torch.save(g_b.state_dict(), ckpt_dir / f"netG_B_ep{epoch}.pth")
            torch.save(d_a.state_dict(), ckpt_dir / f"netD_A_ep{epoch}.pth")
            torch.save(d_b.state_dict(), ckpt_dir / f"netD_B_ep{epoch}.pth")
            # 最新权重（无 epoch 后缀，供 generate.py 默认加载）
            torch.save(g_a.state_dict(), ckpt_dir / "netG_A.pth")
            torch.save(g_b.state_dict(), ckpt_dir / "netG_B.pth")
            print(f"  saved -> {ckpt_dir}")

    print(f"[CycleGAN] 训练完成，权重保存在 {ckpt_dir}")


if __name__ == "__main__":
    train()
