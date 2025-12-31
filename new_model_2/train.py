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
from torch.autograd import Variable

from model import _netG, _netlocalD


# =========================
# 权重初始化
# =========================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview')
    parser.add_argument('--dataroot', default='dataset/train')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nBottleneck', type=int, default=512)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--wtl2', type=float, default=0.9)
    parser.add_argument('--outf', default='result')
    parser.add_argument('--use_amp', action='store_true')
    opt = parser.parse_args()

    print(opt)
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # =========================
    # 随机种子
    # =========================
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    # =========================
    # 数据
    # =========================
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        drop_last=True
    )

    # =========================
    # 模型
    # =========================
    netG = _netG(opt)
    netD = _netlocalD(opt)

    netG.apply(weights_init)
    netD.apply(weights_init)

    if opt.cuda:
        netG.cuda()
        netD.cuda()

    # =========================
    # Loss & Optim
    # =========================
    criterionGAN = nn.MSELoss()
    criterionL1 = nn.L1Loss()

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    # =========================
    # 训练参数
    # =========================
    center = opt.imageSize // 4
    half = opt.imageSize // 2
    real_label = 1.0
    fake_label = 0.0

    # =========================
    # 训练循环
    # =========================
    for epoch in range(opt.niter):
        for i, (real_cpu, _) in enumerate(dataloader):
            if opt.cuda:
                real_cpu = real_cpu.cuda()

            batch_size = real_cpu.size(0)

            real_center = real_cpu[:, :, center:center+half, center:center+half]

            input_cropped = real_cpu.clone()
            c1 = center + opt.overlapPred
            c2 = center + half - opt.overlapPred
            input_cropped[:, :, c1:c2, c1:c2] = 0

            # =====================
            #  Train D
            # =====================
            optimizerD.zero_grad()

            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                fake_full = netG(input_cropped)
                fake = fake_full[:, :, center:center+half, center:center+half]

                pred_real = netD(real_center)
                lossD_real = criterionGAN(pred_real, torch.ones_like(pred_real))

                pred_fake = netD(fake.detach())
                lossD_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))

                lossD = 0.5 * (lossD_real + lossD_fake)

            scaler.scale(lossD).backward()
            scaler.step(optimizerD)

            # =====================
            #  Train G
            # =====================
            optimizerG.zero_grad()

            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                pred_fake = netD(fake)
                lossG_GAN = criterionGAN(pred_fake, torch.ones_like(pred_fake))
                lossG_L1 = criterionL1(fake, real_center)
                lossG = (1 - opt.wtl2) * lossG_GAN + opt.wtl2 * lossG_L1

            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()

            if i % 50 == 0:
                print(f"[{epoch}/{opt.niter}] [{i}/{len(dataloader)}] "
                      f"D: {lossD.item():.4f} | "
                      f"G_GAN: {lossG_GAN.item():.4f} | "
                      f"G_L1: {lossG_L1.item():.4f}")

        # =====================
        # 保存可视化
        # =====================
        recon = input_cropped.clone()
        recon[:, :, center:center+half, center:center+half] = fake
        vutils.save_image((recon+1)/2, f"{opt.outf}/recon_epoch_{epoch:03d}.png")

        torch.save({'epoch': epoch, 'state_dict': netG.state_dict()},
                   f"model/netG_epoch_{epoch}.pth")

    print("Training finished.")


if __name__ == '__main__':
    main()
