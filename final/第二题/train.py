from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import _netlocalD, _netG

# 可选：添加高斯噪声
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='height / width of input images')
    parser.add_argument('--nz', type=int, default=100, help='size of latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--netG', default='', help="path to netG to continue training")
    parser.add_argument('--netD', default='', help="path to netD to continue training")
    parser.add_argument('--outf', default='.', help='output folder')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nBottleneck', type=int, default=4000)
    parser.add_argument('--overlapPred', type=int, default=4)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--wtl2', type=float, default=0.998)
    parser.add_argument('--wtlD', type=float, default=0.001)
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision training')
    opt = parser.parse_args()
    print(opt)

    # 创建文件夹
    for folder in ["result/train/cropped", "result/train/real", "result/train/recon", "model"]:
        os.makedirs(folder, exist_ok=True)

    # 随机种子
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: CUDA device detected, consider running with --cuda")

    # ---------------- 数据集 ----------------
    transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.imageSize, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    if opt.dataset in ['imagenet', 'folder', 'lfw', 'streetview']:
        dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'], transform=transform)
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    # ---------------- 模型 ----------------
    ngpu, nz, ngf, ndf, nc, nef = int(opt.ngpu), int(opt.nz), int(opt.ngf), int(opt.ndf), 3, int(opt.nef)
    nBottleneck, wtl2 = int(opt.nBottleneck), float(opt.wtl2)
    overlapL2Weight = 10

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    resume_epoch = 0
    netG = _netG(opt)
    netG.apply(weights_init)
    if opt.netG != '':
        checkpointG = torch.load(opt.netG, map_location=lambda storage, loc: storage)
        netG.load_state_dict(checkpointG['state_dict'])
        resume_epoch = checkpointG['epoch']

    netD = _netlocalD(opt)
    netD.apply(weights_init)
    if opt.netD != '':
        checkpointD = torch.load(opt.netD, map_location=lambda storage, loc: storage)
        netD.load_state_dict(checkpointD['state_dict'])
        resume_epoch = checkpointD['epoch']

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    label = torch.FloatTensor(opt.batchSize, 1)
    real_label, fake_label = 1, 0
    real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize // 2, opt.imageSize // 2)

    if opt.cuda:
        netD, netG = netD.cuda(), netG.cuda()
        criterion, criterionMSE = criterion.cuda(), criterionMSE.cuda()
        input_real, input_cropped, real_center, label = \
            input_real.cuda(), input_cropped.cuda(), real_center.cuda(), label.cuda()

    input_real, input_cropped, label, real_center = \
        Variable(input_real), Variable(input_cropped), Variable(label), Variable(real_center)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    scaler = torch.amp.GradScaler(enabled=opt.use_amp)

    # ---------------- 训练 ----------------
    center = opt.imageSize // 4
    half = opt.imageSize // 2

    # 更新 scaler
    scaler = torch.amp.GradScaler(enabled=opt.use_amp)

    for epoch in range(resume_epoch, opt.niter):
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
            batch_size = real_cpu.size(0)

            real_center_cpu = real_cpu[:, :, center:center + half, center:center + half]

            input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
            input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
            real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

            # 填充 cropped 中心
            input_cropped.data[:, 0, center + opt.overlapPred:center + half - opt.overlapPred,
            center + opt.overlapPred:center + half - opt.overlapPred] = 2 * 117.0 / 255.0 - 1.0
            input_cropped.data[:, 1, center + opt.overlapPred:center + half - opt.overlapPred,
            center + opt.overlapPred:center + half - opt.overlapPred] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2, center + opt.overlapPred:center + half - opt.overlapPred,
            center + opt.overlapPred:center + half - opt.overlapPred] = 2 * 123.0 / 255.0 - 1.0

            # --------- 训练 D ---------
            netD.zero_grad()
            label.data.fill_(real_label)
            with torch.amp.autocast(device_type='cuda', enabled=opt.use_amp):
                output = netD(real_center)
                errD_real = criterion(output, label)
            scaler.scale(errD_real).backward()
            D_x = output.data.mean()

            fake = netG(input_cropped)
            label.data.fill_(fake_label)
            with torch.amp.autocast(device_type='cuda', enabled=opt.use_amp):
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
            scaler.scale(errD_fake).backward()
            D_G_z1 = output.data.mean()
            optimizerD.step()

            # --------- 训练 G ---------
            netG.zero_grad()
            label.data.fill_(real_label)
            with torch.amp.autocast(device_type='cuda', enabled=opt.use_amp):
                output = netD(fake)
                errG_D = criterion(output, label)

                wtl2Matrix = real_center.clone()
                wtl2Matrix.data.fill_(wtl2 * 10)
                wtl2Matrix.data[:, :, opt.overlapPred:half - opt.overlapPred,
                opt.overlapPred:half - opt.overlapPred] = wtl2
                errG_l2 = ((fake - real_center).pow(2) * wtl2Matrix).mean()
                errG = (1 - wtl2) * errG_D + wtl2 * errG_l2

            scaler.scale(errG).backward()
            D_G_z2 = output.data.mean()
            scaler.step(optimizerG)
            scaler.update()

            if i % 100 == 0:
                print(f'[{epoch}/{opt.niter}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD_real.item()+errD_fake.item():.4f} '
                      f'Loss_G: {errG_D.item():.4f}/{errG_l2.item():.4f} '
                      f'l_D(x): {D_x:.4f} l_D(G(z)): {D_G_z1:.4f}')

        # 保存结果图像
        vutils.save_image(real_cpu, f'result/train/real/real_samples_epoch_{epoch:03d}.png')
        vutils.save_image(input_cropped.data, f'result/train/cropped/cropped_samples_epoch_{epoch:03d}.png')
        recon_image = input_cropped.clone()
        recon_image.data[:, :, center:center+half, center:center+half] = fake.data
        vutils.save_image(recon_image.data, f'result/train/recon/recon_center_samples_epoch_{epoch:03d}.png')

        # 保存模型
        torch.save({'epoch': epoch+1, 'state_dict': netG.state_dict()}, f'model/netG_epoch_{epoch+1}.pth')
        torch.save({'epoch': epoch+1, 'state_dict': netD.state_dict()}, f'model/netlocalD_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    main()
