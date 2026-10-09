import torch
import torch.nn as nn


class ModalityAlignmentLoss(nn.Module):
    """MOS 的 class-wise 模态对齐损失（CMAL）。

        L_CMAL = (1/|C|) * sum_c ( ||mu_opt^c - mu_sar^c||_2^2
                                   + ||var_opt^c - var_sar^c||_2^2 )

    其中 mu / var 为该身份在对应模态下的特征质心与逐维方差，全部取**批内**统计。

    实现约定
    --------
    - 相机：camid == 0 为光学，camid == 1 为 SAR（与 HOSS 数据集一致）；
    - 特征先转 float32 再 L2 归一化，避免 fp16 下逐维方差的精度崩塌；
    - 身份在两个模态都有样本时才计入，n_pair 为 0 时返回 0。

    已实测的性质（不要再为此"修复"）
    -------------------------------
    本赛题 2828 个训练身份中，2662 个只有 1 张光学图、2782 个只有 1 张 SAR 图，
    两个模态同时 >= 2 张的身份仅 44 个（1.56%）。NUM_INSTANCE=2 强制每个身份在
    batch 内恰好 2 个样本，因此批内每个模态恒为 1 个样本：单样本方差恒为 0，且该
    0 与特征取值无关、梯度也恒为 0 —— **方差项实际上从未参与优化**，本损失退化为
    纯质心对齐。曾尝试用跨 batch 类统计记忆库激活方差项（logs/SDF-Net-mosmem），
    实测 Failed：Final 仅 +0.0016，fold0 -0.0007 / fold1 +0.0039，两折不同号且增益
    全部来自 38 条已饱和的 O2O，按铁律（min(fold) >= 0.003）已删除。
    """

    def __init__(self, opt_cam=0, sar_cam=1, eps=1e-6):
        super().__init__()
        self.opt_cam = opt_cam
        self.sar_cam = sar_cam
        self.eps = eps

    def forward(self, feat, pids, camids):
        if feat is None or feat.numel() == 0:
            return torch.zeros((), device=camids.device)

        device = feat.device
        f = torch.nn.functional.normalize(feat.float(), p=2, dim=1, eps=self.eps)

        loss = torch.zeros((), device=device)
        n_pair = 0

        for pid in torch.unique(pids):
            sel = pids == pid
            m_opt = sel & (camids == self.opt_cam)
            m_sar = sel & (camids == self.sar_cam)
            if not bool(m_opt.any()) or not bool(m_sar.any()):
                continue

            f_opt = f[m_opt]
            f_sar = f[m_sar]
            # 质心项
            loss = loss + (f_opt.mean(dim=0) - f_sar.mean(dim=0)).pow(2).sum()
            # 方差项（批内统计；NUM_INSTANCE=2 时恒为 0，见 docstring）
            var_opt = f_opt.var(dim=0, unbiased=False)
            var_sar = f_sar.var(dim=0, unbiased=False)
            loss = loss + (var_opt - var_sar).pow(2).sum()
            n_pair += 1

        if n_pair == 0:
            return torch.zeros((), device=device)
        return loss / n_pair