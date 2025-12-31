# training.py - 修复版本，添加进度条
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import sys


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # 损失函数和优化器
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # 训练记录
        self.best_val_dice = 0.0

        # 创建保存目录
        os.makedirs(config.model_save_dir, exist_ok=True)

        # 早停
        '''self.patience = 10  # 耐心值
        self.best_loss = float('inf')
        self.early_stop_counter = 0'''

    def train_epoch(self, epoch):
        """训练一个epoch - 添加进度条"""
        self.model.train()
        epoch_loss = 0.0
        total_batches = len(self.train_loader)

        print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            # 进度条
            progress = (batch_idx + 1) / total_batches * 100
            sys.stdout.write(f"\rProgress: [{batch_idx + 1}/{total_batches}] {progress:.1f}%")
            sys.stdout.flush()

            images = images.to(self.device)
            targets = targets.to(self.device)

            # one-hot转类别索引（2通道->单通道）
            targets = torch.argmax(targets, dim=1).long()  # [B, 2, H, W] -> [B, H, W]

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)  # 输出应为[B, 2, H, W]
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        sys.stdout.write("\n")  # 换行
        return epoch_loss / total_batches

    def dice_score(self, pred, target, smooth=1e-6):
        """计算Dice系数"""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        # 展平
        pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean().item()

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        val_loss = 0.0

        print("Validating...")

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)

        # 计算Dice（抽样计算，加快速度）
        with torch.no_grad():
            sample_images, sample_targets, _ = next(iter(self.val_loader))
            sample_images = sample_images.to(self.device)
            sample_targets = sample_targets.to(self.device)
            sample_outputs = self.model(sample_images)
            val_dice = self.dice_score(sample_outputs, sample_targets)

        # 更新学习率
        self.scheduler.step(avg_val_loss)

        # 保存最佳模型
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            self.save_checkpoint(epoch, is_best=True)

        return avg_val_loss, val_dice

    def train(self):
        """完整训练流程"""
        print(f"\nStarting training...")
        print(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 50)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss, val_dice = self.validate(epoch)

            # 显示epoch结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Dice: {val_dice:.4f} | "
                  f"LR: {current_lr:.6f}")

            # 定期保存
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)

            # 每5个epoch显示一次时间
            if (epoch + 1) % 5 == 0:
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (epoch + 1) * self.config.epochs
                remaining = estimated_total - elapsed_time
                print(f"  Time elapsed: {elapsed_time / 60:.1f}min, "
                      f"Estimated remaining: {remaining / 60:.1f}min")

        # 保存最终模型
        self.save_checkpoint(self.config.epochs - 1, is_final=True)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.1f} minutes")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")

    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }

        if is_best:
            filename = f'unet_best_dice{self.best_val_dice:.4f}.pth'
        elif is_final:
            filename = 'unet_final.pth'
        else:
            filename = f'unet_epoch{epoch + 1}.pth'

        save_path = os.path.join(self.config.model_save_dir, filename)
        torch.save(checkpoint, save_path)

        if is_best:
            print(f"  Saved best model: {filename}")
