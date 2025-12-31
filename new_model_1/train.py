from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import _netlocalD, _netG

# =========================================================
# 强制开启 AMP（GPU + torch 2.7.1）
# =========================================================
USE_AMP = True


@torch.no_grad()
def compute_metrics(fake, real):
    """
    fake, real: [-1,1]
    """
    fake = (fake + 1) / 2
    real = (real + 1) / 2

    l1 = torch.mean(torch.abs(fake - real))
    l2 = torch.mean((fake - real) ** 2)
    psnr = 10 * torch.log10(1.0 / (l2 + 1e-8))

    return l1.item(), l2.item(), psnr.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview')
    parser.add_argument('--dataroot', default='dataset/train')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nBottleneck', type=int, default=1024)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--wtl2', type=float, default=0.998)
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
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    dataset = dset.ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        persistent_workers=True
    )

    # ================= Model =================
    netG = _netG(opt).to(device)
    netD = _netlocalD(opt).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizerG = optim.Adam(netG.parameters(), opt.lr, (opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), opt.lr, (opt.beta1, 0.999))

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    center = opt.imageSize // 4
    half = opt.imageSize // 2

    # ================= Best Model Tracker =================
    best_psnr = -1.0

    # ================= Training =================
    for epoch in range(opt.niter):
        netG.train()
        netD.train()

        sum_l1 = sum_l2 = sum_psnr = 0.0
        cnt = 0

        for real_cpu, _ in dataloader:
            real_cpu = real_cpu.to(device, non_blocking=True)
            real_center = real_cpu[:, :, center:center + half, center:center + half]

            input_cropped = real_cpu.clone()
            input_cropped[:, :,
                center + opt.overlapPred:center + half - opt.overlapPred,
                center + opt.overlapPred:center + half - opt.overlapPred
            ] = 0

            # -------- Train D --------
            optimizerD.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                out_real = netD(real_center)
                loss_real = criterion(out_real, torch.ones_like(out_real))

                fake = netG(input_cropped)
                out_fake = netD(fake.detach())
                loss_fake = criterion(out_fake, torch.zeros_like(out_fake))

                loss_D = loss_real + loss_fake

            scaler.scale(loss_D).backward()
            scaler.step(optimizerD)

            # -------- Train G --------
            optimizerG.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                out_fake = netD(fake)
                loss_adv = criterion(out_fake, torch.ones_like(out_fake))
                l2 = (fake - real_center).pow(2).mean()
                loss_G = (1 - opt.wtl2) * loss_adv + opt.wtl2 * l2

            scaler.scale(loss_G).backward()
            scaler.step(optimizerG)
            scaler.update()

            # -------- Metrics --------
            with torch.no_grad():
                l1, l2v, psnr = compute_metrics(fake, real_center)

            sum_l1 += l1
            sum_l2 += l2v
            sum_psnr += psnr
            cnt += 1

        avg_psnr = sum_psnr / cnt

        print(
            f"[Epoch {epoch+1}/{opt.niter}] "
            f"L1: {sum_l1/cnt:.4f} "
            f"L2: {sum_l2/cnt:.5f} "
            f"PSNR: {avg_psnr:.2f}"
        )

        # ================= 强制每 epoch 存图 =================
        vutils.save_image(
            fake.detach(),
            f"result/train/recon/recon_{epoch:03d}.png",
            normalize=True
        )

        # ================= Save latest =================
        torch.save({
            "epoch": epoch + 1,
            "psnr": avg_psnr,
            "state_dict": netG.state_dict()
        }, "model/netG_latest.pth")

        # ================= Save best =================
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                "epoch": epoch + 1,
                "psnr": best_psnr,
                "state_dict": netG.state_dict()
            }, "model/netG_best.pth")
            print(f"New best model saved! PSNR = {best_psnr:.2f}")


if __name__ == "__main__":
    main()
