import traceback
from tkinter import Image

import torch
import cv2
import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw  # 原代码缺少此导入
from PIL.ImageFont import ImageFont
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from ultralytics.nn.modules import MorphologicalPreprocess


def visualize_morphological_processing(
        input_dir: str,
        output_dir: str,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: tuple = (1536, 96),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    形态学预处理可视化函数

    参数：
    - input_dir: 输入图像目录
    - output_dir: 输出目录
    - img_size: 输入图像尺寸 (W, H)
    - device: 使用的计算设备
    """
    # 初始化预处理模块
    model = MorphologicalPreprocess(in_channels, out_channels).to(device).eval()

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 处理所有图像文件
    for img_path in Path(input_dir).glob("*.*"):
        if img_path.suffix.lower() in [".png", ".jpg", ".bmp"]:
            try:
                # 读取并预处理图像
                orig_img = cv2.imread(str(img_path))
                if orig_img is None:
                    continue

                # 调整尺寸并转换为Tensor
                resized_img = cv2.resize(orig_img, img_size)
                input_tensor = ToTensor()(resized_img).unsqueeze(0).to(device)

                # 可视化不同阶段的处理结果
                with torch.no_grad():
                    # 原始形态学操作结果
                    morph_feat = model.morphological_operations(input_tensor)

                    # 最终输出结果
                    output = model(input_tensor)

                # 可视化形态学特征
                visualize_features(morph_feat[0], output_dir, img_path.stem)

                # 可视化最终输出
                save_output(output[0], output_dir, img_path.name)

                # 可视化原始对比图
                save_comparison(orig_img, resized_img, output[0], output_dir, img_path.name)

            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {str(e)}")


def visualize_features(feature_tensor, output_dir, filename):
    """可视化形态学特征图"""
    # 分解不同处理阶段的特征
    features = {
        "horizontal": feature_tensor[3:12].cpu(),
        "vertical": feature_tensor[12:21].cpu(),
        "dots": feature_tensor[21:30].cpu()
    }

    # 保存各通道特征图
    for name, feat in features.items():
        for ch in range(3):
            channel = feat[ch].numpy()
            normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(Path(output_dir) / f"{filename}_{name}_ch{ch}.png"), normalized)

            # 修复融合图生成逻辑
            merged = np.zeros((feat.shape[1], feat.shape[2], 3), dtype=np.uint8)  # 预分配uint8类型数组
        for ch in range(3):
            # 独立归一化并转换类型
            channel = cv2.normalize(feat[ch].numpy(), None, 0, 255, cv2.NORM_MINMAX)
            merged[..., ch] = channel.astype(np.uint8)

        # 生成有效的伪彩色图
        try:
            # 方案1：转换为灰度图后应用色图
            gray = cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            cv2.imwrite(str(Path(output_dir) / f"{filename}_{name}_heatmap.png"), heatmap)

            # 方案2：直接保存融合图（可选）
            cv2.imwrite(str(Path(output_dir) / f"{filename}_{name}_merged.png"), merged)

        except Exception as e:
            print(f"生成伪彩色图时出错：{str(e)}")
            traceback.print_exc()


def save_output(output_tensor, output_dir, filename):
    """保存最终输出图像"""
    output = output_tensor.cpu().numpy().transpose(1, 2, 0)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(Path(output_dir) / f"processed_{filename}"), output)


def save_comparison(orig, resized, output, output_dir, filename):
    """生成对比图 (修复尺寸处理问题)"""
    # 统一所有图像的尺寸为原始图像尺寸
    target_size = orig.shape[1], orig.shape[0]  # (width, height)

    # 转换原始图像 (BGR → RGB)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig_pil = Image.fromarray(orig_rgb)

    # 调整resized图像到原始尺寸
    resized_resized = cv2.resize(resized, target_size)
    resized_pil = Image.fromarray(cv2.cvtColor(resized_resized, cv2.COLOR_BGR2RGB))

    # 处理输出图像
    output_np = output.cpu().numpy().transpose(1, 2, 0)
    output_normalized = cv2.normalize(output_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    output_resized = cv2.resize(output_normalized, target_size)
    output_pil = Image.fromarray(output_resized)

    # 创建对比图
    total_width = orig_pil.width * 3
    comparison = Image.new("RGB", (total_width, orig_pil.height))

    # 拼接图像
    comparison.paste(orig_pil, (0, 0))
    comparison.paste(resized_pil, (orig_pil.width, 0))
    comparison.paste(output_pil, (orig_pil.width * 2, 0))

    # 添加标注
    draw = ImageDraw.Draw(comparison)
    # font = ImageFont.load_default()
    draw.text((10, 10), "Original", fill=(255, 0, 0))
    draw.text((orig_pil.width + 10, 10), "Resized", fill=(255, 0, 0))
    draw.text((orig_pil.width * 2 + 10, 10), "Processed", fill=(255, 0, 0))

    comparison.save(Path(output_dir) / f"comparison_{filename}")


if __name__ == "__main__":
    # 使用示例
    visualize_morphological_processing(
        input_dir="D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/MP_test",
        output_dir="D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/datasets/morph_visualization_ours",
        img_size=(3072, 96),  # 根据实际需求调整
        in_channels=3,
        out_channels=3
    )