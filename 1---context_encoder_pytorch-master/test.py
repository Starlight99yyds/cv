from __future__ import print_function
import argparse
import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import _netG
import numpy as np
from psnr import psnr
import torchvision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview', help='dataset type')
    parser.add_argument('--dataroot', default='dataset/val', help='dataset path')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--netG', default='model/netG_epoch_498.pth') #netG_streetview 498_10.07
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--nBottleneck', type=int, default=4000)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision inference')
    opt = parser.parse_args()
    print(opt)

    # ---------------- 模型 ----------------
    netG = _netG(opt)
    checkpoint = torch.load(opt.netG, map_location=lambda storage, loc: storage)
    netG.load_state_dict(checkpoint['state_dict'])
    netG.eval()
    if opt.cuda:
        netG.cuda()

    # ---------------- 数据 ----------------
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root=opt.dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=opt.workers)

    input_cropped = torch.FloatTensor(opt.batchSize,3,opt.imageSize,opt.imageSize)
    real_center = torch.FloatTensor(opt.batchSize,3,opt.imageSize//2,opt.imageSize//2)
    if opt.cuda:
        input_cropped = input_cropped.cuda()
        real_center = real_center.cuda()
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)

    os.makedirs('result', exist_ok=True)

    # ---------------- 指标初始化 ----------------
    total_l1, total_l2, total_psnr = 0, 0, 0
    num_images = 0
    alpha = 0.9  # 边界融合比例
    center = opt.imageSize // 4
    half = opt.imageSize // 2

    # ---------------- 循环数据集 ----------------
    for batch_idx, (real_cpu, _) in enumerate(dataloader):
        batch_size = real_cpu.size(0)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center_cpu = real_cpu[:, :, center:center+half, center:center+half]
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

        # 填充中心区域
        c_start = center + opt.overlapPred
        c_end = center + half - opt.overlapPred
        input_cropped.data[:,0,c_start:c_end,c_start:c_end] = 2*117.0/255-1.0
        input_cropped.data[:,1,c_start:c_end,c_start:c_end] = 2*104.0/255-1.0
        input_cropped.data[:,2,c_start:c_end,c_start:c_end] = 2*123.0/255-1.0

        # ---------------- 前向 ----------------
        with torch.amp.autocast(device_type='cuda', enabled=opt.use_amp):
            fake = netG(input_cropped)

        # ---------------- 边界融合 ----------------
        recon_image = input_cropped.clone()
        center_slice = slice(center, center+half)
        recon_image.data[:, :, center_slice, center_slice] = alpha*fake.data + (1-alpha)*input_cropped.data[:, :, center_slice, center_slice]

        # ---------------- 保存图片 ----------------
        grid = vutils.make_grid((recon_image.data+1)/2.0, nrow=int(math.sqrt(batch_size)))
        vutils.save_image(grid, f'result/recon_batch_{batch_idx}.png')

        # ---------------- 计算指标 ----------------
        fake_np = fake.data.cpu().numpy()
        real_np = real_center.data.cpu().numpy()
        diff = real_np - fake_np
        batch_l2 = np.mean(np.square(diff))
        batch_l1 = np.mean(np.abs(diff))

        # PSNR
        real_disp = (real_np+1)*127.5
        fake_disp = (fake_np+1)*127.5
        batch_psnr = 0
        for i in range(batch_size):
            batch_psnr += psnr(real_disp[i].transpose(1,2,0), fake_disp[i].transpose(1,2,0))
        batch_psnr /= batch_size

        total_l1 += batch_l1 * batch_size
        total_l2 += batch_l2 * batch_size
        total_psnr += batch_psnr * batch_size
        num_images += batch_size

        print(f"Batch {batch_idx}: L1={batch_l1:.4f}, L2={batch_l2:.4f}, PSNR={batch_psnr:.2f}")

    # ---------------- 平均指标 ----------------
    print("\n===== Method Evaluation =====")
    print(f"Mean L1 Loss: {total_l1/num_images:.4f}")
    print(f"Mean L2 Loss: {total_l2/num_images:.4f}")
    print(f"PSNR (higher better): {total_psnr/num_images:.2f}")


if __name__ == "__main__":
    main()
