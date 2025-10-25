import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
from glob import glob
from ultralytics import YOLO


# 特征图捕获器
class FeatureHook:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def hook_fn(self, module, input, output):
        self.inputs = input[0].detach().cpu()
        self.outputs = output.detach().cpu()


# 注册钩子
def register_hooks(model, layer_names):
    hooks = {}
    for name, module in model.named_modules():
        for key, layer_name in layer_names.items():
            if name == layer_name:
                hook = FeatureHook()
                module.register_forward_hook(hook.hook_fn)
                hooks[key] = hook
                print(f"成功注册钩子到层：{layer_name}")  # 添加调试打印
    return hooks


# 新增融合热力图可视化函数
# 修改后的融合热力图可视化函数
def visualize_fused_features(features, save_path):
    if features is None:
        print(f"警告：特征数据为空，无法可视化")
        return

    os.makedirs(save_path, exist_ok=True)

    # 计算通道平均值
    fused = torch.mean(features, dim=1)  # [B, H, W]

    # 对比度增强
    fused = (fused - torch.quantile(fused, 0.02)) / (torch.quantile(fused, 0.98) - torch.quantile(fused, 0.02) + 1e-8)
    fused = torch.clamp(fused, 0, 1)

    # 获取特征图原始尺寸
    feat_height, feat_width = fused.shape[1], fused.shape[2]

    # 设置与特征图尺寸匹配的画布
    dpi = 100
    fig = plt.figure(
        figsize=(feat_width / dpi, feat_height / dpi),  # 精确匹配特征图尺寸
        dpi=dpi,
        frameon=False
    )
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 保持原始像素尺寸显示
    ax.imshow(fused[0].numpy(), cmap='jet', aspect='auto')

    # 保存与特征图尺寸完全一致的热力图
    plt.savefig(
        os.path.join(save_path, 'fused_heatmap.png'),
        format='png',
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close()

    # 验证保存尺寸
    saved_img = plt.imread(os.path.join(save_path, 'fused_heatmap.png'))
    assert saved_img.shape[:2] == (feat_height, feat_width), \
        f"保存尺寸不匹配！期望({feat_height},{feat_width}), 实际{saved_img.shape[:2]}"


# 可视化函数（保存单独通道图片）
# 可视化函数（合并通道+高宽比保持+高清输出）
def visualize_features(features, save_path):
    """多通道RGB融合可视化"""
    if features is None or features.numel() == 0:
        print(f"特征数据无效，跳过保存路径：{save_path}")
        return

    # 确保输入为4D张量 [B, C, H, W]
    if features.dim() == 3:
        features = features.unsqueeze(0)
    B, C, H, W = features.shape

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # === 全通道融合逻辑 ===
    # 将通道均匀分配到RGB三通道（若C不足3则循环填充）
    r_weights = torch.zeros(C)
    g_weights = torch.zeros(C)
    b_weights = torch.zeros(C)

    for c in range(C):
        # 循环分配权重（示例：均匀分配）
        if c % 3 == 0:
            r_weights[c] = 1.0
        elif c % 3 == 1:
            g_weights[c] = 1.0
        else:
            b_weights[c] = 1.0

    # 归一化权重
    r_weights /= r_weights.sum() + 1e-8
    g_weights /= g_weights.sum() + 1e-8
    b_weights /= b_weights.sum() + 1e-8

    # 生成RGB分量
    with torch.no_grad():
        r_channel = torch.einsum('bchw,c->bhw', features, r_weights)
        g_channel = torch.einsum('bchw,c->bhw', features, g_weights)
        b_channel = torch.einsum('bchw,c->bhw', features, b_weights)

    # === 动态归一化 ===
    def normalize(channel):
        channel_np = channel.cpu().numpy()
        vmin = np.percentile(channel_np, 2)
        vmax = np.percentile(channel_np, 98)
        return np.clip((channel_np - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 分量归一化
    r_norm = normalize(r_channel)
    g_norm = normalize(g_channel)
    b_norm = normalize(b_channel)

    # === 保存单通道分量 ===
    def save_single_channel(data, path):
        plt.imsave(path, data, cmap='gray', vmin=0, vmax=1)
        print(f"保存单通道图：{os.path.basename(path)}")

    # 保存R分量
    save_single_channel(r_norm[0], os.path.join(save_path, "R_channel.png"))
    # 保存G分量
    save_single_channel(g_norm[0], os.path.join(save_path, "G_channel.png"))
    # 保存B分量
    save_single_channel(b_norm[0], os.path.join(save_path, "B_channel.png"))

    # === 生成并保存RGB融合图 ===
    rgb_image = np.stack([r_norm[0], g_norm[0], b_norm[0]], axis=-1)
    plt.imsave(os.path.join(save_path, "RGB_fusion.png"), rgb_image)
    print(f"保存RGB融合图：RGB_fusion.png")

    # === 验证输出尺寸 ===
    for fname in ["R_channel.png", "G_channel.png", "B_channel.png", "RGB_fusion.png"]:
        img = plt.imread(os.path.join(save_path, fname))
        assert img.shape[:2] == (H, W), f"尺寸错误：{fname} 期望({H},{W})，实际{img.shape[:2]}"


# 单张图片处理流程
def process_single_image(img_path, save_subdir, model, hooks, layer_names):
    try:
        # 读取原始图像（直接从文件读取，使用PIL库）
        import cv2
        orig_img_cv = cv2.imread(img_path)
        if orig_img_cv is None:
            raise ValueError("无法读取图片")

        # 保存原始图像（OpenCV直接保存，颜色正确）
        cv2.imwrite(f"{save_subdir}/00_original.png", orig_img_cv)

        # 运行推理
        with torch.no_grad():
            model.to('cuda')
            results = model.predict(img_path, verbose=False)
            model.to('cpu')

        # 新增：处理推理结果（示例：保存带检测框的图像）
        if results:
            for result in results:
                annotated_img = result.plot()  # 生成带检测框的图像
                cv2.imwrite(
                    f"{save_subdir}/detected_with_boxes.png",
                    cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)  # 转换颜色空间
                )

        # 可视化各模块前后的特征图
        for key, layer_name in layer_names.items():
            hook = hooks.get(key)
            if hook:
                # 创建模块专属保存目录
                module_dir = os.path.join(save_subdir, key)
                os.makedirs(module_dir, exist_ok=True)

                # # 保存输入特征图
                # input_dir = os.path.join(module_dir, 'input')
                # visualize_features(hook.inputs, f"Before {layer_name}", input_dir)
                #
                # # 保存输出特征图
                # output_dir = os.path.join(module_dir, 'output')
                # visualize_features(hook.outputs, f"After {layer_name}", output_dir)

                # 正确参数顺序（关键修复）
                visualize_features(hook.inputs, os.path.join(module_dir, 'input'))
                visualize_features(hook.outputs, os.path.join(module_dir, 'output'))

                # 新增融合热力图可视化
                visualize_fused_features(hook.inputs, os.path.join(module_dir, 'input'))
                visualize_fused_features(hook.outputs, os.path.join(module_dir, 'output'))

        del results
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"处理 {img_path} 时出错: {e}")


# 单张图片处理流程
# def process_single_image(img_path, save_subdir, model, hooks, layer_names):
#     try:
#         # 读取原始图像（直接从文件读取，使用PIL库）
#         import cv2
#         orig_img_cv = cv2.imread(img_path)
#         if orig_img_cv is None:
#             raise ValueError("无法读取图片")
#
#         # 保存原始图像（OpenCV直接保存，颜色正确）
#         cv2.imwrite(f"{save_subdir}/00_original.png", orig_img_cv)
#
#         # 运行推理
#         with torch.no_grad():
#             model.to('cuda')
#             results = model.predict(img_path, verbose=False)
#             model.to('cpu')
#
#         # 新增：处理推理结果（示例：保存带检测框的图像）
#         if results:
#             for result in results:
#                 annotated_img = result.plot()  # 生成带检测框的图像
#                 cv2.imwrite(
#                     f"{save_subdir}/detected_with_boxes.jpg",
#                     cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)  # 转换颜色空间
#                 )
#
#         # 可视化各模块前后的特征图
#         for key, layer_name in layer_names.items():
#             hook = hooks.get(key)
#             if hook:
#                 visualize_features(hook.inputs,
#                                    f"Before {layer_name}",
#                                    f"{save_subdir}/{key}_input.png")
#                 visualize_features(hook.outputs,
#                                    f"After {layer_name}",
#                                    f"{save_subdir}/{key}_output.png")
#
#         del results
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     except Exception as e:
#         print(f"处理 {img_path} 时出错: {e}")


# 主流程
if __name__ == "__main__":
    # 配置参数
    model_path = "D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/ours_2max/weights/last.pt"  # 替换为你的模型路径
    input_dir = "D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/MP_test/"  # 替换为你的输入图像文件夹路径
    save_dir = "D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/visualize_activation_stats_ours_2max/"  # 替换为你的结果保存目录
    layer_names = {
        'preprocess': 'model.0',
        'msecsp': 'model.8'
    }

    # 初始化模型
    model = YOLO(model_path)
    model.model.eval()

    # 在主流程中添加以下代码，打印所有层名
    # pytorch_model = model.model
    # print("所有模型层名：")
    # for name, module in pytorch_model.named_modules():
    #     print(name)

    # 注册钩子
    hooks = register_hooks(model.model, layer_names)

    # 获取所有图片文件
    image_files = glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                  glob(os.path.join(input_dir, "*.[pP][nN][gG]")) + \
                  glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]"))

    batch_size = 1  # 根据显存情况调整
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        # 处理当前批次
        # 遍历处理每张图片
        for img_file in image_files:
            # 创建保存子目录
            img_name = Path(img_file).stem
            save_subdir = os.path.join(save_dir, img_name)
            os.makedirs(save_subdir, exist_ok=True)

            # 处理单张图片
            print(f"Processing: {img_file}")
            process_single_image(img_file, save_subdir, model, hooks, layer_names)
        # 批次处理后清理显存
        torch.cuda.empty_cache()
        gc.collect()

    print(f"所有可视化结果已保存至 {save_dir} 目录")

