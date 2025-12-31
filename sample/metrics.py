import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    def __init__(self, num_classes=2, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _preprocess(self, pred, target):
        """预处理预测和目标"""
        # 确保都是torch张量
        if isinstance(pred, torch.Tensor):
            pred = pred
        else:
            pred = torch.FloatTensor(pred)

        if isinstance(target, torch.Tensor):
            target = target
        else:
            target = torch.FloatTensor(target)

        # 移动到设备
        pred = pred.to(self.device)
        target = target.to(self.device)

        # 将预测通过sigmoid并阈值化
        if pred.dim() == 4 and pred.size(1) > 1:  # [B, C, H, W]
            pred = torch.sigmoid(pred)
            pred = (pred > self.threshold).float()

        return pred, target

    def dice_score(self, pred, target, smooth=1e-6):
        """计算Dice系数（F1分数）"""
        pred, target = self._preprocess(pred, target)

        if pred.dim() == 4:  # [B, C, H, W]
            # 展平
            pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

            intersection = (pred_flat * target_flat).sum(dim=2)
            union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

            dice = (2. * intersection + smooth) / (union + smooth)
            return dice.mean(dim=1).mean().item()
        else:
            # 单个样本
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            return (2. * intersection + smooth) / (union + smooth).item()

    def iou_score(self, pred, target, smooth=1e-6):
        """计算交并比（IoU/Jaccard指数）"""
        pred, target = self._preprocess(pred, target)

        if pred.dim() == 4:
            pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

            intersection = (pred_flat * target_flat).sum(dim=2)
            union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

            iou = (intersection + smooth) / (union + smooth)
            return iou.mean(dim=1).mean().item()
        else:
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            return (intersection + smooth) / (union + smooth).item()

    def precision_score(self, pred, target, smooth=1e-6):
        """计算精确率"""
        pred, target = self._preprocess(pred, target)

        if pred.dim() == 4:
            pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

            true_positives = (pred_flat * target_flat).sum(dim=2)
            predicted_positives = pred_flat.sum(dim=2)

            precision = (true_positives + smooth) / (predicted_positives + smooth)
            return precision.mean(dim=1).mean().item()
        else:
            true_positives = (pred * target).sum()
            predicted_positives = pred.sum()
            return (true_positives + smooth) / (predicted_positives + smooth).item()

    def recall_score(self, pred, target, smooth=1e-6):
        """计算召回率"""
        pred, target = self._preprocess(pred, target)

        if pred.dim() == 4:
            pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

            true_positives = (pred_flat * target_flat).sum(dim=2)
            actual_positives = target_flat.sum(dim=2)

            recall = (true_positives + smooth) / (actual_positives + smooth)
            return recall.mean(dim=1).mean().item()
        else:
            true_positives = (pred * target).sum()
            actual_positives = target.sum()
            return (true_positives + smooth) / (actual_positives + smooth).item()

    def specificity_score(self, pred, target, smooth=1e-6):
        """计算特异性（真阴性率）"""
        pred, target = self._preprocess(pred, target)

        if pred.dim() == 4:
            pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

            # 真阴性：预测为负且实际为负
            true_negatives = ((1 - pred_flat) * (1 - target_flat)).sum(dim=2)
            # 实际阴性
            actual_negatives = (1 - target_flat).sum(dim=2)

            specificity = (true_negatives + smooth) / (actual_negatives + smooth)
            return specificity.mean(dim=1).mean().item()
        else:
            true_negatives = ((1 - pred) * (1 - target)).sum()
            actual_negatives = (1 - target).sum()
            return (true_negatives + smooth) / (actual_negatives + smooth).item()

    def hausdorff_distance(self, pred, target):
        """计算豪斯多夫距离（近似）"""
        pred, target = self._preprocess(pred, target)

        # 简化的豪斯多夫距离计算
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        # 计算两个集合之间的最大最小距离
        if pred_binary.sum() == 0 or target_binary.sum() == 0:
            return float('inf')

        return 0.0  # 简化的实现

    def compute_all_metrics(self, pred, target):
        """计算所有指标"""
        metrics = {}
        metrics['dice'] = self.dice_score(pred, target)
        metrics['iou'] = self.iou_score(pred, target)
        metrics['precision'] = self.precision_score(pred, target)
        metrics['recall'] = self.recall_score(pred, target)
        metrics['specificity'] = self.specificity_score(pred, target)

        # 计算F1分数（与Dice相同但重新计算以确保）
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        return metrics

    def print_metrics_report(self, metrics, class_names=['LV', 'RV', 'Myocardium']):
        """打印指标报告"""
        print("\n" + "=" * 60)
        print("SEGMENTATION PERFORMANCE METRICS REPORT")
        print("=" * 60)

        # 总体指标
        print(f"\nOverall Metrics:")
        print(f"{'Metric':<20} {'Value':<10}")
        print(f"{'-' * 30}")
        print(f"{'Dice Score':<20} {metrics['dice']:.4f}")
        print(f"{'IoU Score':<20} {metrics['iou']:.4f}")
        print(f"{'Precision':<20} {metrics['precision']:.4f}")
        print(f"{'Recall':<20} {metrics['recall']:.4f}")
        print(f"{'Specificity':<20} {metrics['specificity']:.4f}")
        print(f"{'F1 Score':<20} {metrics['f1']:.4f}")

        print("\n" + "=" * 60)