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
    dist = torch.sqrt(x ** 2 + y ** 2)
    return torch.clamp(1 - dist, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset/val')
    parser.add_argument('--batchSize', type=int, default=16)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--netG', default='model/netG_epoch_93.pth')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nBottleneck', type=int, default=512)
    opt = parser.parse_args()

    os.makedirs('result_test', exist_ok=True)
    device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    # =========================
    # 模型
    # =========================
    netG = _netG(opt)
    checkpoint = torch.load(opt.netG, map_location='cpu')
    netG.load_state_dict(checkpoint['state_dict'])
    netG.to(device)
    netG.eval()

    # =========================
    # 数据
    # =========================
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    dataset = torchvision.datasets.ImageFolder(opt.dataroot, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    center = opt.imageSize // 4
    half = opt.imageSize // 2
    mask = cosine_mask(half, device).unsqueeze(0).unsqueeze(0)

    total_psnr = 0
    count = 0
    all_recon = []

    # =========================
    # 推理
    # =========================
    with torch.no_grad():
        for idx, (real_cpu, _) in enumerate(dataloader):
            real_cpu = real_cpu.to(device)
            batch_size = real_cpu.size(0)

            input_cropped = real_cpu.clone()
            c1 = center + opt.overlapPred
            c2 = center + half - opt.overlapPred
            input_cropped[:, :, c1:c2, c1:c2] = 0

            fake_full = netG(input_cropped)
            fake = fake_full[:, :, center:center + half, center:center + half]
            real_center = real_cpu[:, :, center:center + half, center:center + half]

            recon = input_cropped.clone()
            recon[:, :, center:center + half, center:center + half] = fake * mask + real_center * (1 - mask)

            # -----------------------------
            # 保存单张对比图： Masked | Recon | GT
            # -----------------------------
            for i in range(batch_size):
                masked_img = (input_cropped[i:i+1] + 1) / 2
                recon_img = (recon[i:i+1] + 1) / 2
                gt_img = (real_cpu[i:i+1] + 1) / 2
                compare_img = torch.cat([masked_img, recon_img, gt_img], dim=3)
                vutils.save_image(compare_img,
                                  f"result_test/compare_{idx*batch_size + i:03d}.png")

            # -----------------------------
            # 收集所有 Recon，用于拼图
            # -----------------------------
            all_recon.append((recon + 1) / 2)

            # -----------------------------
            # PSNR
            # -----------------------------
            fake_np = ((fake + 1) / 2).cpu().numpy() * 255
            real_np = ((real_center + 1) / 2).cpu().numpy() * 255
            for i in range(fake_np.shape[0]):
                total_psnr += psnr(fake_np[i], real_np[i])
                count += 1

    print(f"Average PSNR: {total_psnr / count:.2f} dB")

    # =========================
    # 保存 8x8 和 6x6 拼图（只用 Recon）
    # =========================
    all_recon = torch.cat(all_recon, dim=0)
    assert all_recon.size(0) >= 100, "测试集数量不足 100"

    vutils.save_image(all_recon[:64],
                      "result_test/recon_8x8.png",
                      nrow=8,
                      padding=2)

    vutils.save_image(all_recon[64:100],
                      "result_test/recon_6x6.png",
                      nrow=6,
                      padding=2)


if __name__ == '__main__':
    main()
