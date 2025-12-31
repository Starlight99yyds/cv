# test.py - 医学图像分割模型测试脚本
import torch
import numpy as np
from PIL import Image
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 在导入matplotlib之前设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from model.UNet import UNet
from dataset import MedicalSegmentationDataset


class Tester:
    def __init__(self, model_path, test_data_dir):
        """初始化测试器

        Args:
            model_path: 模型权重文件路径
            test_data_dir: 测试数据目录
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 获取模型配置
        in_channels = checkpoint['config'].in_channels
        out_channels = checkpoint['config'].out_channels

        print(f"Model config - in_channels: {in_channels}, out_channels: {out_channels}")

        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model: {model_path}")
        print(f"Best Dice from training: {checkpoint['best_val_dice']:.4f}")

        # 加载测试集
        self.test_dataset = MedicalSegmentationDataset(
            os.path.join(test_data_dir, 'val/images'),
            os.path.join(test_data_dir, 'val/labels')
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )

        # 保存输出通道数
        self.out_channels = out_channels

        # 创建结果目录
        self.results_dir = './test_results'
        os.makedirs(self.results_dir, exist_ok=True)

        # 获取当前时间
        self.test_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def dice_score(self, pred, target, smooth=1e-6):
        """计算Dice系数

        Args:
            pred: 预测张量 [B, C, H, W]
            target: 目标张量 [B, C, H, W]
            smooth: 平滑项防止除零

        Returns:
            dice系数
        """
        # 处理预测：对于二分类，取第二个通道作为前景
        if pred.size(1) == 2:
            # 取前景通道（索引1）
            pred = (pred[:, 1:2, :, :] > 0.5).float()
        else:
            pred = (pred > 0.5).float()

        # 确保目标与预测维度一致
        if target.size(1) == 2 and pred.size(1) == 1:
            # 如果目标也是2通道，取对应的前景通道
            target = target[:, 1:2, :, :]
        elif target.size(1) != pred.size(1):
            # 如果维度不匹配，确保目标也是单通道
            target = target[:, 0:1, :, :]

        # 展平张量
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean(dim=1).mean().item()

    def iou_score(self, pred, target, smooth=1e-6):
        """计算IoU（Jaccard指数）

        Args:
            pred: 预测张量 [B, C, H, W]
            target: 目标张量 [B, C, H, W]
            smooth: 平滑项防止除零

        Returns:
            IoU系数
        """
        # 处理预测：对于二分类，取第二个通道作为前景
        if pred.size(1) == 2:
            # 取前景通道（索引1）
            pred = (pred[:, 1:2, :, :] > 0.5).float()
        else:
            pred = (pred > 0.5).float()

        # 确保目标与预测维度一致
        if target.size(1) == 2 and pred.size(1) == 1:
            target = target[:, 1:2, :, :]
        elif target.size(1) != pred.size(1):
            target = target[:, 0:1, :, :]

        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.mean(dim=1).mean().item()

    def calculate_precision_recall(self, pred, target, smooth=1e-6):
        """计算精确率和召回率

        Args:
            pred: 预测张量
            target: 目标张量

        Returns:
            精确率和召回率
        """
        # 处理预测：对于二分类，取第二个通道作为前景
        if pred.size(1) == 2:
            pred = (pred[:, 1:2, :, :] > 0.5).float()
        else:
            pred = (pred > 0.5).float()

        # 确保目标与预测维度一致
        if target.size(1) == 2 and pred.size(1) == 1:
            target = target[:, 1:2, :, :]
        elif target.size(1) != pred.size(1):
            target = target[:, 0:1, :, :]

        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)

        # 精确率 = TP / (TP + FP)
        precision = (intersection + smooth) / (pred_flat.sum(dim=2) + smooth)

        # 召回率 = TP / (TP + FN)
        recall = (intersection + smooth) / (target_flat.sum(dim=2) + smooth)

        return precision.mean(dim=1).mean().item(), recall.mean(dim=1).mean().item()

    def test_all(self):
        """测试整个测试集

        Returns:
            平均Dice和IoU分数
        """
        print(f"\nTesting on {len(self.test_dataset)} images...")
        print("-" * 70)

        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        image_names = []

        with torch.no_grad():
            for i, (images, targets, img_paths) in enumerate(self.test_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                preds = torch.sigmoid(outputs)

                dice = self.dice_score(preds, targets)
                iou = self.iou_score(preds, targets)
                precision, recall = self.calculate_precision_recall(preds, targets)

                dice_scores.append(dice)
                iou_scores.append(iou)
                precision_scores.append(precision)
                recall_scores.append(recall)
                image_names.append(os.path.basename(img_paths[0]))

                print(f"Image {i + 1:2d}/{len(self.test_dataset)}: {image_names[-1]:25s} "
                      f"Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # 计算平均指标
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        std_dice = np.std(dice_scores)
        std_iou = np.std(iou_scores)

        # 打印结果
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Number of test images:     {len(dice_scores)}")
        print(f"Average Dice Score:        {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"Average IoU Score:         {avg_iou:.4f} ± {std_iou:.4f}")
        print(f"Average Precision:         {avg_precision:.4f}")
        print(f"Average Recall:            {avg_recall:.4f}")
        print(f"Best Dice Score:           {np.max(dice_scores):.4f}")
        print(f"Worst Dice Score:          {np.min(dice_scores):.4f}")
        print("-" * 70)

        # 计算F1分数
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)
        print(f"Average F1 Score:          {avg_f1:.4f}")

        # 打印每张图像的结果
        print("\nDetailed Results:")
        print("-" * 70)
        for i, (name, dice, iou, prec, rec) in enumerate(
                zip(image_names, dice_scores, iou_scores, precision_scores, recall_scores)):
            f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
            print(
                f"{i + 1:2d}. {name:25s} Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        print("=" * 70)

        # 保存结果到文件
        self.save_test_results(image_names, dice_scores, iou_scores,
                               precision_scores, recall_scores,
                               avg_dice, avg_iou, avg_precision, avg_recall)

        # 生成结果可视化
        self.plot_results(image_names, dice_scores, iou_scores)

        return avg_dice, avg_iou

    def save_test_results(self, image_names, dice_scores, iou_scores,
                          precision_scores, recall_scores,
                          avg_dice, avg_iou, avg_precision, avg_recall):
        """保存测试结果到文件"""
        results_file = os.path.join(self.results_dir, 'test_results.txt')

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("MEDICAL IMAGE SEGMENTATION TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Date: {self.test_time}\n")
            f.write(f"Model: {os.path.basename(self.model_path) if hasattr(self, 'model_path') else 'UNet'}\n")
            f.write(f"Number of test images: {len(dice_scores)}\n")
            f.write(f"Best Dice from training: 0.9509\n\n")

            f.write("Summary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Dice Score:        {avg_dice:.4f}\n")
            f.write(f"Average IoU Score:         {avg_iou:.4f}\n")
            f.write(f"Average Precision:         {avg_precision:.4f}\n")
            f.write(f"Average Recall:            {avg_recall:.4f}\n")
            f.write(
                f"Average F1 Score:          {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6):.4f}\n\n")

            f.write("Detailed Results:\n")
            f.write("-" * 70 + "\n")
            f.write("No.  Image Name                   Dice     IoU      Precision Recall   F1\n")
            f.write("-" * 70 + "\n")

            for i, (name, dice, iou, prec, rec) in enumerate(
                    zip(image_names, dice_scores, iou_scores, precision_scores, recall_scores)):
                f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
                f.write(f"{i + 1:2d}.  {name:25s} {dice:.4f}   {iou:.4f}   {prec:.4f}    {rec:.4f}    {f1:.4f}\n")

            f.write("=" * 70 + "\n")

        print(f"\nResults saved to: {results_file}")

    def plot_results(self, image_names, dice_scores, iou_scores):
        """生成结果可视化图表"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        x = np.arange(len(image_names))

        # Dice分数条形图
        axes[0].bar(x, dice_scores, color='skyblue', alpha=0.7, label='Dice Score')
        axes[0].axhline(y=np.mean(dice_scores), color='red', linestyle='--',
                        label=f'Average: {np.mean(dice_scores):.4f}')
        axes[0].set_xlabel('Image Index')
        axes[0].set_ylabel('Dice Score')
        axes[0].set_title('Dice Scores for Test Images')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'{i + 1}' for i in x])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # IoU分数条形图
        axes[1].bar(x, iou_scores, color='lightgreen', alpha=0.7, label='IoU Score')
        axes[1].axhline(y=np.mean(iou_scores), color='red', linestyle='--',
                        label=f'Average: {np.mean(iou_scores):.4f}')
        axes[1].set_xlabel('Image Index')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_title('IoU Scores for Test Images')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{i + 1}' for i in x])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'scores_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Scores distribution plot saved to: {plot_path}")

    def test_single(self, image_path, gt_path):
        """测试单张图像

        Args:
            image_path: 输入图像路径
            gt_path: ground truth路径

        Returns:
            预测结果、Dice分数、IoU分数
        """
        # 保存模型路径用于结果记录
        if not hasattr(self, 'model_path'):
            self.model_path = image_path

        # 加载图像
        image = Image.open(image_path).convert('L')
        gt = Image.open(gt_path).convert('L')

        # 获取原始图像和标签
        image_np = np.array(image)
        gt_np = np.array(gt)

        # 预处理
        image_tensor = torch.FloatTensor(image_np).unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.sigmoid(output)

        print(f"Prediction shape: {pred.shape}")

        # 处理ground truth - 二分类情况
        gt_binary = (gt_np > 0).astype(np.float32)
        gt_tensor = torch.FloatTensor(gt_binary).unsqueeze(0).unsqueeze(0).to(self.device)

        print(f"Ground truth shape: {gt_tensor.shape}")

        # 计算指标
        dice = self.dice_score(pred, gt_tensor)
        iou = self.iou_score(pred, gt_tensor)
        precision, recall = self.calculate_precision_recall(pred, gt_tensor)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # 获取前景预测
        pred_foreground = pred[:, 1:2, :, :] > 0.5
        pred_np = pred_foreground.squeeze().cpu().numpy()

        # 统计信息
        pred_foreground_pixels = np.sum(pred_np > 0)
        gt_foreground_pixels = np.sum(gt_np > 0)
        intersection_pixels = np.sum((pred_np > 0) & (gt_np > 0))
        union_pixels = np.sum((pred_np > 0) | (gt_np > 0))

        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Dice Score:          {dice:.4f}")
        print(f"IoU Score:           {iou:.4f}")
        print(f"Precision:           {precision:.4f}")
        print(f"Recall:              {recall:.4f}")
        print(f"F1 Score:            {f1:.4f}")
        print(f"\nPixel Statistics:")
        print(f"Prediction foreground:  {pred_foreground_pixels}")
        print(f"Ground truth foreground: {gt_foreground_pixels}")
        print(f"Intersection:           {intersection_pixels}")
        print(f"Union:                  {union_pixels}")
        print(f"Pixel difference:       {abs(pred_foreground_pixels - gt_foreground_pixels)}")
        print(f"Pixel overlap ratio:    {intersection_pixels / union_pixels:.4f}")

        # 保存可视化结果
        self.save_visualization(image_np, gt_np, pred_np, image_path,
                                dice, iou, precision, recall)

        return pred.cpu(), dice, iou

    def save_visualization(self, image_np, gt_np, pred_np, image_path,
                           dice, iou, precision, recall):
        """保存可视化结果"""
        vis_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 创建对比图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：原始图像、GT、预测
        axes[0, 0].imshow(image_np, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth', fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(pred_np, cmap='gray')
        axes[0, 2].set_title('Prediction', fontsize=12)
        axes[0, 2].axis('off')

        # 第二行：重叠、差异、指标
        # 重叠图像
        overlap = np.zeros((*pred_np.shape, 3))
        overlap[..., 0] = pred_np * 0.8  # 红色表示预测
        overlap[..., 1] = gt_np * 0.8  # 绿色表示GT
        overlap = np.clip(overlap, 0, 1)
        axes[1, 0].imshow(overlap)
        axes[1, 0].set_title('Overlap (Red: Pred, Green: GT)', fontsize=12)
        axes[1, 0].axis('off')

        # 差异图像
        diff = np.abs(pred_np - gt_np)
        axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title('Difference (Pred - GT)', fontsize=12)
        axes[1, 1].axis('off')

        # 指标文本
        axes[1, 2].axis('off')
        metrics_text = f'Evaluation Metrics:\n\n'
        metrics_text += f'Dice Score:     {dice:.4f}\n'
        metrics_text += f'IoU Score:      {iou:.4f}\n'
        metrics_text += f'Precision:      {precision:.4f}\n'
        metrics_text += f'Recall:         {recall:.4f}\n'
        metrics_text += f'F1 Score:       {2 * (precision * recall) / (precision + recall + 1e-6):.4f}\n\n'
        metrics_text += f'Image: {image_name}'

        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11,
                        verticalalignment='center', transform=axes[1, 2].transAxes,
                        family='monospace')

        plt.suptitle(f'Medical Image Segmentation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'{image_name}_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {save_path}")

        # 保存预测结果为图像
        pred_save_path = os.path.join(vis_dir, f'{image_name}_prediction.png')
        pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_img.save(pred_save_path)
        print(f"Prediction mask saved to: {pred_save_path}")


def main():
    """主函数"""
    # 测试配置
    model_path = './checkpoints/unet_final - 报告.pth'
    test_data_dir = './data'

    # 创建测试器
    tester = Tester(model_path, test_data_dir)

    # 测试整个测试集
    print("\n" + "=" * 70)
    print("FULL TEST SET EVALUATION")
    print("=" * 70)
    avg_dice, avg_iou = tester.test_all()

    # 测试单张图像（示例）
    print("\n" + "=" * 70)
    print("SINGLE IMAGE ANALYSIS")
    print("=" * 70)

    #sample_img = './data/val/images/CHNCXR_0660_1.png'
    #sample_gt = './data/val/labels/CHNCXR_0660_1.png'
    sample_img = './data/val/images/CHNCXR_0001_0.png'
    sample_gt = './data/val/labels/CHNCXR_0001_0.png'

    if os.path.exists(sample_img) and os.path.exists(sample_gt):
        print(f"\nTesting sample image: {os.path.basename(sample_img)}")
        print("-" * 70)
        pred, dice, iou = tester.test_single(sample_img, sample_gt)
    else:
        print(f"Sample image not found at {sample_img}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score:  {avg_iou:.4f}")
    print("=" * 70)

    return avg_dice, avg_iou


if __name__ == '__main__':
    main()