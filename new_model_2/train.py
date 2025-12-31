from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torchvision.models import vgg16
from model import _netlocalD, _netG

# =========================================================
# 强制 AMP
# =========================================================
USE_AMP = True


# ================= VGG Perceptual =================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # [-1,1] → [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.mean(torch.abs(x - y))
        return loss


@torch.no_grad()
def compute_psnr(fake, real):
    fake = (fake + 1) / 2
    real = (real + 1) / 2
    mse = torch.mean((fake - real) ** 2)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview')
    parser.add_argument('--dataroot', default='dataset/train')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--niter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--nBottleneck', type=int, default=4000)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--wtl2', type=float, default=0.998)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    opt = parser.parse_args()
    print(opt)

    os.makedirs("result/train/recon", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # ================= Seed =================
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    device = torch.device("cuda")

    # ================= Dataset =================
    transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.imageSize, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    dataset = dset.ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    # ================= Model =================
    netG = _netG(opt).to(device)
    netD = _netlocalD(opt).to(device)

    criterion_adv = nn.BCEWithLogitsLoss().to(device)
    criterion_l2 = nn.MSELoss().to(device)
    perceptual = VGGPerceptualLoss().to(device)

    optimizerG = optim.Adam(netG.parameters(), opt.lr, (opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), opt.lr, (opt.beta1, 0.999))

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    center = opt.imageSize // 4
    half = opt.imageSize // 2

    best_psnr = -1.0

    # ================= Training =================
    for epoch in range(opt.niter):
        netG.train()
        netD.train()

        psnr_sum = 0.0
        cnt = 0

        for real_cpu, _ in dataloader:
            real_cpu = real_cpu.to(device)
            real_center = real_cpu[:, :, center:center+half, center:center+half]

            # ---- masked input（使用 ImageNet mean，而不是 0）----
            input_cropped = real_cpu.clone()
            input_cropped[:, 0,
                center+opt.overlapPred:center+half-opt.overlapPred,
                center+opt.overlapPred:center+half-opt.overlapPred] = 2*117/255 - 1
            input_cropped[:, 1,
                center+opt.overlapPred:center+half-opt.overlapPred,
                center+opt.overlapPred:center+half-opt.overlapPred] = 2*104/255 - 1
            input_cropped[:, 2,
                center+opt.overlapPred:center+half-opt.overlapPred,
                center+opt.overlapPred:center+half-opt.overlapPred] = 2*123/255 - 1

            # ========== Train D ==========
            optimizerD.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out_real = netD(real_center)
                loss_real = criterion_adv(out_real, torch.ones_like(out_real))

                fake = netG(input_cropped)
                out_fake = netD(fake.detach())
                loss_fake = criterion_adv(out_fake, torch.zeros_like(out_fake))

                loss_D = loss_real + loss_fake

            scaler.scale(loss_D).backward()
            scaler.step(optimizerD)

            # ========== Train G ==========
            optimizerG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out_fake = netD(fake)
                loss_adv = criterion_adv(out_fake, torch.ones_like(out_fake))
                loss_l2 = criterion_l2(fake, real_center)
                loss_vgg = perceptual(fake, real_center)

                loss_G = (
                    0.001 * loss_adv +
                    opt.wtl2 * loss_l2 +
                    0.1 * loss_vgg
                )

            scaler.scale(loss_G).backward()
            scaler.step(optimizerG)
            scaler.update()

            with torch.no_grad():
                psnr_sum += compute_psnr(fake, real_center)
                cnt += 1

        avg_psnr = psnr_sum / cnt
        print(f"[Epoch {epoch+1}] PSNR: {avg_psnr:.2f}")

        # ================= 强制存图 =================
        vutils.save_image(
            fake.detach(),
            f"result/train/recon/recon_{epoch:03d}.png",
            normalize=True
        )

        torch.save(netG.state_dict(), "model/netG_latest.pth")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(netG.state_dict(), "model/netG_best.pth")
            print(f"✓ Best model saved (PSNR={best_psnr:.2f})")


if __name__ == "__main__":
    main()
