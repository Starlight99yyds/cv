# train.py - GPU版本，添加进度显示
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
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载配置
    config = get_config()

    # 创建模型并移动到设备
    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    ).to(device)  # 将模型移动到GPU

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

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 设置为0以避免多进程问题
        pin_memory=True,  # 当使用GPU时，pin_memory可以提高数据加载速度
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,  # 当使用GPU时，pin_memory可以提高数据加载速度
        persistent_workers=False
    )

    # 检查第一批数据
    print("Checking first batch of data...")
    for images, targets, names in train_loader:
        # 将数据也移动到GPU
        images = images.to(device)
        targets = targets.to(device)
        print(f"Batch shape - Images: {images.shape}, Targets: {targets.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        print(f"Data device: Images on {images.device}, Targets on {targets.device}")
        break

    print("-" * 50)

    # 创建训练器并开始训练 - 传递device参数
    trainer = Trainer(model, train_loader, val_loader, config)
    # Trainer类内部已经处理了device，这里不需要再传递

    trainer.train()


if __name__ == '__main__':
    main()