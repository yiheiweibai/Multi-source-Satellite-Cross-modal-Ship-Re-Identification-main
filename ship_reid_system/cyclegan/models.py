"""CycleGAN 模型：ResNet 生成器 + PatchGAN 判别器。

适配 256x256 光学/SAR 跨模态翻译。
SAR 为三通道灰度复制形式，optical 为真彩三通道，因此直接用 3->3 通道。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str = "reflect", norm_layer=nn.InstanceNorm2d,
                 use_dropout: bool = False, use_bias: bool = True):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based 生成器：下采样 -> n_blocks 残差 -> 上采样。"""

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64,
                 n_blocks: int = 9, norm_layer=nn.InstanceNorm2d, use_dropout: bool = False,
                 padding_type: str = "reflect"):
        super().__init__()
        use_bias = True
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        # 下采样
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]
        # 残差块
        mult = 2 ** 2
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]
        # 上采样
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN 判别器，输出 70x70 patch 的真假概率图。"""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    """LSGAN 损失（更稳定）。"""

    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return self.real_label.expand_as(prediction)
        return self.fake_label.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


def build_cyclegan(ngf: int = 64, ndf: int = 64, n_blocks: int = 9):
    """返回 G_A(A->B), G_B(B->A), D_A(B 域判别器), D_B(A 域判别器)。

    训练约定（见 train.py）：A=optical, B=sar，故 G_A=optical->sar, G_B=sar->optical。
    """
    norm = nn.InstanceNorm2d
    g_a = ResnetGenerator(3, 3, ngf, n_blocks, norm)
    g_b = ResnetGenerator(3, 3, ngf, n_blocks, norm)
    d_a = NLayerDiscriminator(3, ndf, 3, norm)
    d_b = NLayerDiscriminator(3, ndf, 3, norm)
    return g_a, g_b, d_a, d_b
