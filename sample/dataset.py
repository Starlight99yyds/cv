# dataset.py - 二分类版本
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MedicalSegmentationDataset(Dataset):
    def __init__(self, img_dir, gt_dir, augment=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        # 获取文件列表
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

        print(f"数据集初始化: {len(self.img_files)} 个样本")
        print(f"图像目录: {img_dir}")
        print(f"标签目录: {gt_dir}")

        # 转换
        self.transform = transforms.ToTensor()

        # 数据增强
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)

        image = Image.open(img_path).convert('L')  # 灰度图
        gt = Image.open(gt_path).convert('L')

        # 转换为numpy数组进行调试
        image_np = np.array(image)
        gt_np = np.array(gt)

        # 调试信息（可选，查看前几个样本）
        '''if idx < 3:
            print(f"\n样本 {idx} ({img_name}):")
            print(f"  图像范围: [{image_np.min()}, {image_np.max()}]")
            print(f"  标签范围: [{gt_np.min()}, {gt_np.max()}]")
            print(f"  标签唯一值: {np.unique(gt_np)}")'''

        # 关键修改：处理标签
        # 你的标签是0和255，需要转换为0和1
        if gt_np.max() > 1:  # 如果是0-255
            gt_np = (gt_np > 128).astype(np.float32)  # 阈值128，>128的转为1

        # 转换为Tensor
        image_tensor = self.transform(image).float()

        # 归一化图像到[0,1]
        if image_tensor.max() > 1:
            image_tensor = image_tensor / 255.0

        # 对于二分类，有两种处理方式：
        # 方式1：单通道输出 + sigmoid
        # gt_tensor = torch.FloatTensor(gt_np).unsqueeze(0)  # [1, H, W]

        # 方式2：两通道one-hot输出 + softmax（推荐）
        gt_onehot = np.zeros((2, gt_np.shape[0], gt_np.shape[1]), dtype=np.float32)
        gt_onehot[0] = (gt_np == 0).astype(np.float32)  # 背景通道
        gt_onehot[1] = (gt_np == 1).astype(np.float32)  # 前景通道（肺）

        gt_tensor = torch.FloatTensor(gt_onehot)

        return image_tensor, gt_tensor, img_name