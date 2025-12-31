import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# =========================
# U-Net Generator
# =========================
class _netG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        nf = opt.nef

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True)
            )

        self.d1 = down(opt.nc, nf)
        self.d2 = down(nf, nf*2)
        self.d3 = down(nf*2, nf*4)
        self.d4 = down(nf*4, nf*8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf*8, opt.nBottleneck, 4, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.nBottleneck, nf*8, 4, 1, 0),
            nn.ReLU(True)
        )

        self.u4 = up(nf*16, nf*4)
        self.u3 = up(nf*8, nf*2)
        self.u2 = up(nf*4, nf)
        self.u1 = nn.ConvTranspose2d(nf*2, opt.nc, 4, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        b = self.bottleneck(d4)

        u4 = self.u4(torch.cat([b, d4], 1))
        u3 = self.u3(torch.cat([u4, d3], 1))
        u2 = self.u2(torch.cat([u3, d2], 1))
        out = torch.tanh(self.u1(torch.cat([u2, d1], 1)))
        return out


# =========================
# PatchGAN Discriminator
# =========================
class _netlocalD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ndf

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(opt.nc, nf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf, nf*2, 4, 2, 1)),
            nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf*2, nf*4, 4, 2, 1)),
            nn.InstanceNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf*4, nf*8, 4, 1, 1)),
            nn.InstanceNorm2d(nf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nf*8, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.main(x)
