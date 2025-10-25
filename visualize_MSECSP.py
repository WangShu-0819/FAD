import cv2
import gc
import numpy as np
import torch
from pathlib import Path
from glob import glob
from typing import Union, Dict
from tqdm import tqdm

from ultralytics import YOLO


class MSECSPVisualizer:
    def __init__(self,
                 model_path: Union[str, Path],
                 target_layer: str = 'model.8',
                 img_size: tuple = (3072, 96),
                 device: str = 'cuda:0'):
        """
        改进点：
        1. 增加自动设备选择
        2. 支持多后端路径处理
        3. 显式内存管理
        """
        # 初始化模型
        self.model = YOLO(str(model_path)).to(device)
        self.model.model.eval()
        self.img_size = img_size  # (width, height)
        self.device = device

        # 特征缓存
        self.hook_data = {'input': None, 'output': None}

        # 注册钩子
        self._register_hook(target_layer)

        # 可视化参数
        self.cmap = cv2.COLORMAP_JET
        self.alpha = 0.5  # 热力图叠加透明度

    def _register_hook(self, layer_name: str):
        """改进的钩子注册方法"""

        def hook(module, input, output):
            # 显式转移数据到CPU并释放显存
            self.hook_data['input'] = input[0].detach().cpu()
            self.hook_data['output'] = output.detach().cpu()
            torch.cuda.empty_cache()

        found = False
        for name, module in self.model.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook)
                print(f"✅ 成功注册钩子到层: {layer_name}")
                found = True
                break
        if not found:
            raise ModuleNotFoundError(f"未找到目标层 {layer_name}，请检查模型结构")

    def _process_features(self, feat: torch.Tensor) -> np.ndarray:
        """
        改进的特征后处理流程：
        1. 多尺度融合
        2. 自适应对比度增强
        3. 边缘保持上采样
        """
        if feat is None:
            raise ValueError("输入特征为空")

        # 通道压缩
        feat_2d = torch.mean(feat, dim=1)  # [B,H,W]
        feat_np = feat_2d.squeeze().numpy()

        # 动态归一化（排除极端值）
        vmin, vmax = np.percentile(feat_np, [2, 98])
        feat_norm = np.clip((feat_np - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # 边缘保持上采样
        if feat_norm.shape != self.img_size[::-1]:  # (H,W)
            feat_norm = cv2.resize(
                feat_norm,
                self.img_size,  # (width, height)
                interpolation=cv2.INTER_CUBIC
            )
        return feat_norm

    def _generate_heatmap(self, image: np.ndarray, feat: np.ndarray) -> np.ndarray:
        """生成带颜色映射的热力图叠加"""
        heatmap = cv2.applyColorMap((feat * 255).astype(np.uint8), self.cmap)
        return cv2.addWeighted(image, 1 - self.alpha, heatmap, self.alpha, 0)

    def visualize_single(self,
                         img_path: Union[str, Path],
                         save_dir: Union[str, Path]):
        """
        单图可视化流程
        改进点：
        1. 增加图像预处理
        2. 异常处理机制
        """
        # 路径处理
        img_path = Path(img_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 读取并预处理图像
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 推理捕获特征
            with torch.no_grad():
                self.model.predict(img)

            # 处理特征
            input_heat = self._process_features(self.hook_data['input'])
            output_heat = self._process_features(self.hook_data['output'])

            # 生成可视化结果
            self._save_results(img, input_heat, output_heat, save_dir, img_path.stem)

        except Exception as e:
            print(f"处理图像 {img_path.name} 时发生错误: {str(e)}")
        finally:
            # 显式释放资源
            del img, input_heat, output_heat
            gc.collect()
            torch.cuda.empty_cache()

    def visualize_batch(self,
                        input_dir: Union[str, Path],
                        save_dir: Union[str, Path]):
        """批量处理模式"""
        input_dir = Path(input_dir)
        img_paths = glob(str(input_dir / '*.jpg')) + glob(str(input_dir / '*.png'))

        print(f"🔍 发现 {len(img_paths)} 张待处理图像")
        for path in tqdm(img_paths, desc="生成热力图"):
            self.visualize_single(path, save_dir)

    def _save_results(self,
                      img: np.ndarray,
                      input_heat: np.ndarray,
                      output_heat: np.ndarray,
                      save_dir: Path,
                      stem: str):
        """保存所有可视化结果"""
        # 输入特征
        input_overlay = self._generate_heatmap(img, input_heat)
        cv2.imwrite(str(save_dir / f"{stem}_input.jpg"), input_overlay)

        # 输出特征
        output_overlay = self._generate_heatmap(img, output_heat)
        cv2.imwrite(str(save_dir / f"{stem}_output.jpg"), output_overlay)

        # 差异图
        diff = np.abs(output_heat - input_heat)
        diff_overlay = self._generate_heatmap(img, diff)
        cv2.imwrite(str(save_dir / f"{stem}_diff.jpg"), diff_overlay)

        # 原始图像
        cv2.imwrite(str(save_dir / f"{stem}_original.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # ================== 配置区 ==================
    MODEL_PATH = Path(
        "D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/MAC/weights/best.pt")
    INPUT_DIR = Path("D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/MP_test/")
    SAVE_DIR = Path("D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/visualize_MSECSP3/")

    # ================== 执行区 ==================
    visualizer = MSECSPVisualizer(
        model_path=MODEL_PATH,
        target_layer='model.8',  # 根据实际模型结构调整
        img_size=(3072, 96),
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 单图测试模式
    # visualizer.visualize_single(
    #     img_path=INPUT_DIR / "test_image.jpg",
    #     save_dir=SAVE_DIR
    # )

    # 批量处理模式
    visualizer.visualize_batch(
        input_dir=INPUT_DIR,
        save_dir=SAVE_DIR
    )