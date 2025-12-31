# train.py - 修复版本，添加进度显示
import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
import time

from configs.config_loader import get_config
from model.UNet import UNet
from dataset import MedicalSegmentationDataset
from model.training import Trainer


def main():
    # 加载配置
    config = get_config()

    # 创建模型
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 加载数据集 - 先检查数据是否存在
    train_img_dir = os.path.join(config.data_dir, 'train/images')
    train_gt_dir = os.path.join(config.data_dir, 'train/labels')

    if not os.path.exists(train_img_dir):
        print(f"Error: Training image directory not found: {train_img_dir}")
        return

    print(f"Loading dataset from {train_img_dir}...")

    # 创建数据集时添加进度显示
    print("Creating dataset...")
    dataset = MedicalSegmentationDataset(train_img_dir, train_gt_dir)
    print(f"Dataset created with {len(dataset)} samples")

    # 划分训练集和验证集
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print("-" * 50)

    # 创建数据加载器 - 使用更少的workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0以避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 检查第一批数据
    print("Checking first batch of data...")
    for images, targets, names in train_loader:
        print(f"Batch shape - Images: {images.shape}, Targets: {targets.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        break

    print("-" * 50)

    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == '__main__':
    main()