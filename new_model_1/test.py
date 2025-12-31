from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import _netG
from psnr import psnr
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default='dataset/val', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='model/netG_best.pth', help='path to generator checkpoint')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nBottleneck', type=int, default=1024, help='of dim for bottleneck of encoder')
    parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
    parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
    parser.add_argument('--wtl2', type=float, default=0.999, help='0 means do not use else use with this weight')
    opt = parser.parse_args()
    print(opt)

    # 设置随机种子
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # 检查模型文件是否存在
    if not os.path.exists(opt.netG):
        raise FileNotFoundError(f"Model file not found: {opt.netG}")

    # 加载模型
    netG = _netG(opt)

    # 加载检查点
    checkpoint = torch.load(opt.netG, map_location=lambda storage, location: storage)

    # 处理不同的checkpoint格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 加载权重（使用strict=False以处理不匹配）
    try:
        netG.load_state_dict(state_dict)
        print("Successfully loaded state_dict with strict=True")
    except RuntimeError as e:
        print(f"Warning: Could not load state_dict with strict=True: {e}")
        print("Trying with strict=False...")
        netG.load_state_dict(state_dict, strict=False)
        print("Loaded state_dict with strict=False (some parameters may not match)")

    netG.eval()

    # 设置设备
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if opt.cuda:
        netG = netG.cuda()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),  # 修复：Scale -> Resize
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
    print(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers)
    )

    # 准备张量
    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    # 修复：整数除法
    center_size = opt.imageSize // 2
    real_center = torch.FloatTensor(opt.batchSize, 3, center_size, center_size)

    criterionMSE = nn.MSELoss()

    # 移动到设备
    if opt.cuda:
        netG = netG.cuda()
        input_real = input_real.cuda()
        input_cropped = input_cropped.cuda()
        real_center = real_center.cuda()
        criterionMSE = criterionMSE.cuda()

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)

    # 获取一个批次的数据
    dataiter = iter(dataloader)
    real_cpu, _ = next(dataiter)  # 修复：使用next()而不是.next()

    # 准备输入
    input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
    input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)

    # 提取中心区域
    center_start = opt.imageSize // 4
    center_end = center_start + center_size
    real_center_cpu = real_cpu[:, :, center_start:center_end, center_start:center_end]
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    # 填充中心区域（论文中的做法）
    c_start = center_start + opt.overlapPred
    c_end = center_end - opt.overlapPred
    input_cropped.data[:, 0, c_start:c_end, c_start:c_end] = 2 * 117.0 / 255.0 - 1.0
    input_cropped.data[:, 1, c_start:c_end, c_start:c_end] = 2 * 104.0 / 255.0 - 1.0
    input_cropped.data[:, 2, c_start:c_end, c_start:c_end] = 2 * 123.0 / 255.0 - 1.0

    # 前向传播
    fake = netG(input_cropped)
    errG = criterionMSE(fake, real_center)

    # 重建完整图像
    recon_image = input_cropped.clone()
    recon_image.data[:, :, center_start:center_end, center_start:center_end] = fake.data

    # 保存图像
    vutils.save_image(real_cpu, 'val_real_samples.png', normalize=True)
    vutils.save_image(input_cropped.data, 'val_cropped_samples.png', normalize=True)
    vutils.save_image(recon_image.data, 'val_recon_samples.png', normalize=True)

    # 计算指标
    fake_np = fake.data.cpu().numpy()
    real_center_np = real_center.data.cpu().numpy()

    t = real_center_np - fake_np
    l2 = np.mean(np.square(t))
    l1 = np.mean(np.abs(t))

    # 转换为[0, 255]范围用于PSNR计算
    real_center_disp = (real_center_np + 1) * 127.5
    fake_disp = (fake_np + 1) * 127.5

    # 计算PSNR
    psnr_sum = 0
    for i in range(opt.batchSize):
        # 调整维度顺序：CHW -> HWC
        real_img = real_center_disp[i].transpose(1, 2, 0)
        fake_img = fake_disp[i].transpose(1, 2, 0)
        psnr_sum += psnr(real_img, fake_img)

    avg_psnr = psnr_sum / opt.batchSize

    # 打印结果
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"MSE (L2) Loss: {l2:.6f}")
    print(f"MAE (L1) Loss: {l1:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")

    # 计算更多统计信息
    print(f"MSE per sample: {errG.item():.6f}")

    # 与论文中的基准比较
    print("\n" + "=" * 50)
    print("COMPARISON WITH PAPER (Paris StreetView)")
    print("=" * 50)
    print("From Context Encoders paper (Table 1):")
    print("- NN-inpainting (HOG features): L1=19.92%, L2=6.92%, PSNR=12.79 dB")
    print("- NN-inpainting (our features): L1=15.10%, L2=4.30%, PSNR=14.70 dB")
    print("- Our Reconstruction (joint): L1=10.33%, L2=2.35%, PSNR=17.59 dB")
    print("\nYour results:")
    print(f"- L1 Loss: {l1 * 100:.2f}%")
    print(f"- L2 Loss: {l2 * 100:.2f}%")
    print(f"- PSNR: {avg_psnr:.2f} dB")


if __name__ == '__main__':
    main()