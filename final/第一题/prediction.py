# prediction_simple.py
# prediction.py - 修复版本
import os
# 强制设置环境变量（必须在所有import之前）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from skimage import measure
import sys

# 添加当前目录到路径
sys.path.append('.')

# 设置中文字体路径
if os.name == 'nt':  # Windows系统
    # 使用系统自带的中文字体
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            matplotlib.font_manager.fontManager.addfont(font_path)
            font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✅ 使用中文字体: {font_name}")
            break
    else:
        print("⚠️ 未找到中文字体，使用英文标题")
        # 如果不使用中文字体，就设置一个不会报错的字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
else:  # Linux/Mac系统
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

try:
    from model.UNet import UNet
    from dataset import MedicalSegmentationDataset
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


class SimplePredictor:
    def __init__(self, model_path):
        """初始化预测器"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            sys.exit(1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型 - CrossEntropyLoss版本 (2通道输出)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = UNet(
            in_channels=1,  # 灰度图像
            out_channels=2  # CrossEntropyLoss训练，2通道输出
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载成功")

    def predict_single_image(self, image_path, gt_path, output_dir):
        """预测单张图像并保存对比图"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('L')
            gt = Image.open(gt_path).convert('L')

            # 转换为Tensor
            image_tensor = torch.FloatTensor(np.array(image)).unsqueeze(0).unsqueeze(0) / 255.0
            gt_np = np.array(gt)

            # 处理标签 (0-255 -> 0-1)
            if gt_np.max() > 1:
                gt_np = (gt_np > 128).astype(np.float32)

            # 模型预测
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                output = self.model(image_tensor)

                # CrossEntropyLoss训练: 使用softmax
                prob = torch.softmax(output, dim=1)
                pred_mask = prob[0, 1].cpu().numpy()  # 前景通道

            # 二值化
            pred_binary = (pred_mask > 0.5).astype(np.float32)

            # 准备显示数据
            img_display = image_tensor[0, 0].cpu().numpy()

            # 创建对比图
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # 1. 原始图像
            axes[0].imshow(img_display, cmap='gray')
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            # 2. 真实标签
            axes[1].imshow(gt_np, cmap='gray')
            axes[1].set_title('真实标签')
            axes[1].axis('off')

            # 3. 模型预测
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title('模型预测')
            axes[2].axis('off')

            # 4. 叠加对比
            axes[3].imshow(img_display, cmap='gray', alpha=0.7)

            # 真实轮廓 (绿色)
            if np.max(gt_np) > 0:
                try:
                    contours = measure.find_contours(gt_np, 0.5)
                    for contour in contours:
                        axes[3].plot(contour[:, 1], contour[:, 0], 'g-', linewidth=1.5, alpha=0.8)
                except:
                    pass

            # 预测轮廓 (红色虚线)
            if np.max(pred_binary) > 0:
                try:
                    contours = measure.find_contours(pred_binary, 0.5)
                    for contour in contours:
                        axes[3].plot(contour[:, 1], contour[:, 0], 'r--', linewidth=1.5, alpha=0.8)
                except:
                    pass

            axes[3].set_title('轮廓对比 (绿:真实, 红:预测)')
            axes[3].axis('off')

            # 添加图例
            import matplotlib.patches as mpatches
            green_patch = mpatches.Patch(color='green', alpha=0.8, label='真实标签')
            red_patch = mpatches.Patch(color='red', alpha=0.8, label='模型预测')
            axes[3].legend(handles=[green_patch, red_patch], loc='upper right')

            plt.tight_layout()

            # 保存图像
            img_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f'pred_{img_name}')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  保存: {save_path}")
            return True

        except Exception as e:
            print(f"  处理失败: {e}")
            return False

    def predict_folder(self, img_dir, gt_dir, output_dir):
        """预测整个文件夹的图像"""
        print(f"\n开始预测...")
        print(f"图像目录: {img_dir}")
        print(f"标签目录: {gt_dir}")
        print(f"输出目录: {output_dir}")

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            print("错误: 输入目录不存在")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像文件列表
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"找到 {len(img_files)} 张图像")

        success_count = 0

        for i, img_file in enumerate(img_files):
            print(f"\n处理 [{i + 1}/{len(img_files)}]: {img_file}")

            img_path = os.path.join(img_dir, img_file)
            gt_path = os.path.join(gt_dir, img_file)

            if not os.path.exists(gt_path):
                print(f"  警告: 标签文件不存在: {gt_path}")
                continue

            success = self.predict_single_image(img_path, gt_path, output_dir)
            if success:
                success_count += 1

        print(f"\n{'=' * 60}")
        print(f"完成!")
        print(f"成功处理: {success_count}/{len(img_files)} 张图像")
        print(f"输出目录: {output_dir}")
        print(f"{'=' * 60}")


def main():
    """主函数"""
    print("胸部X光分割 - 批量预测")
    print("=" * 40)

    # 配置路径
    model_path = './checkpoints/unet_best_dice0.9841.pth'
    #model_path = './checkpoints/unet_final.pth'

    # 如果找不到最佳模型，用第一个.pth文件
    if not os.path.exists(model_path):
        checkpoint_files = [f for f in os.listdir('./checkpoints') if f.endswith('.pth')]
        if checkpoint_files:
            model_path = f'./checkpoints/{checkpoint_files[0]}'
            print(f"使用模型: {model_path}")
        else:
            print("错误: 找不到模型文件")
            return

    # 输入输出目录
    test_paths = [
        ('./data/test/images', './data/test/labels'),
        ('./data/val - 副本/images', './data/val - 副本/labels'),
        ('./test/images', './test/labels'),
        ('./val/images', './val/labels'),
    ]

    # 寻找可用的测试集
    img_dir, gt_dir = None, None
    for img_path, gt_path in test_paths:
        if os.path.exists(img_path):
            img_dir = img_path
            gt_dir = gt_path
            break

    if img_dir is None:
        print("错误: 找不到测试数据集")
        print("请检查以下目录是否存在:")
        for img_path, _ in test_paths:
            print(f"  {img_path}")
        return

    # 输出目录
    output_dir = './predictions_output'

    # 运行预测
    predictor = SimplePredictor(model_path)
    predictor.predict_folder(img_dir, gt_dir, output_dir)


if __name__ == '__main__':
    main()