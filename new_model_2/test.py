from __future__ import print_function
import argparse
import os
import math
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from model import _netG


# =========================
# PSNR
# =========================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


# =========================
# Cosine Mask
# =========================
def cosine_mask(size, device):
    t = torch.linspace(-1, 1, size, device=device)
    x, y = torch.meshgrid(t, t, indexing='ij')
    dist = torch.sqrt(x**2 + y**2)
    return torch.clamp(1 - dist, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset/val')
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--netG', default='model/netG_epoch_38.pth')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nBottleneck', type=int, default=512)
    opt = parser.parse_args()

    os.makedirs('result_test', exist_ok=True)

    # =========================
    # 模型
    # =========================
    netG = _netG(opt)
    checkpoint = torch.load(opt.netG, map_location='cpu')
    netG.load_state_dict(checkpoint['state_dict'])
    netG.eval()

    if opt.cuda:
        netG.cuda()

    # =========================
    # 数据
    # =========================
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    dataset = torchvision.datasets.ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False
    )

    center = opt.imageSize // 4
    half = opt.imageSize // 2

    total_psnr = 0
    count = 0

    mask = cosine_mask(half, 'cuda' if opt.cuda else 'cpu')

    # =========================
    # 推理
    # =========================
    for idx, (real_cpu, _) in enumerate(dataloader):
        if opt.cuda:
            real_cpu = real_cpu.cuda()

        input_cropped = real_cpu.clone()
        c1 = center + opt.overlapPred
        c2 = center + half - opt.overlapPred
        input_cropped[:, :, c1:c2, c1:c2] = 0

        with torch.no_grad():
            fake_full = netG(input_cropped)
            fake = fake_full[:, :, center:center+half, center:center+half]

        real_center = real_cpu[:, :, center:center+half, center:center+half]

        recon = input_cropped.clone()
        recon[:, :, center:center+half, center:center+half] = \
            fake * mask + real_center * (1 - mask)

        vutils.save_image((recon+1)/2,
                          f"result_test/recon_{idx:03d}.png",
                          nrow=int(math.sqrt(real_cpu.size(0))))

        # PSNR
        fake_np = ((fake+1)/2).cpu().numpy() * 255
        real_np = ((real_center+1)/2).cpu().numpy() * 255

        for i in range(fake_np.shape[0]):
            total_psnr += psnr(fake_np[i], real_np[i])
            count += 1

    print(f"Average PSNR: {total_psnr/count:.2f} dB")


if __name__ == '__main__':
    main()
