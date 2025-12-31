import torch
import torch.nn as nn


# =========================
# Channel Attention (替代原 ChannelWiseFullyConnected)
# =========================
class ChannelAttention(nn.Module):
    """
    Channel-wise attention for 1x1 feature map
    等价但比 scaling 强得多，且更稳定
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


# =========================
# Generator
# =========================
class _netG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ngpu = opt.ngpu
        C = opt.nBottleneck

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.nef, opt.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.nef, opt.nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.nef * 2, opt.nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.nef * 4, opt.nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.nef * 8, C, 4, 1, 0, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------- Channel Attention --------
        self.channel_attn = ChannelAttention(C, reduction=8)

        # -------- Low-rank cross-channel mixing --------
        r = 8  # 低秩因子，质量不降
        self.cross_channel = nn.Sequential(
            nn.Conv2d(C, C // r, 1, bias=False),
            nn.BatchNorm2d(C // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // r, C, 1, bias=False),
        )

        self.dropout = nn.Dropout2d(0.5)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(C, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.channel_attn(x)
        x = self.dropout(x)
        x = self.cross_channel(x)
        x = self.decoder(x)
        return x


# =========================
# PatchGAN Discriminator（质量↑，收敛更快）
# =========================
class _netlocalD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ngpu = opt.ngpu

        self.main = nn.Sequential(
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 1, bias=False)
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(out.size(0), -1)
