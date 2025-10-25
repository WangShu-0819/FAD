import cv2
import gc
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Union, Dict, List

from torch.utils.data import DataLoader, Dataset, RandomSampler, sampler
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO
from ultralytics.data import dataset


class FabricDefectDataset(Dataset):
    """自定义织物缺陷数据集加载器"""
    def __init__(self, root_dir: Path, img_size: tuple = (3072, 96)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_files = sorted((root_dir / 'images').glob('*.jpg')) + sorted((root_dir / 'images').glob('*.png'))
        self.label_dir = root_dir / 'labels'

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List]]:
        # 加载图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)  # (width, height)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW

        # 标注加载（增强版）
        labels = []
        label_path = self.label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        # 坐标转换
                        x1 = (x_center - width / 2) * self.img_size[0]
                        y1 = (y_center - height / 2) * self.img_size[1]
                        x2 = (x_center + width / 2) * self.img_size[0]
                        y2 = (y_center + height / 2) * self.img_size[1]
                        labels.append({
                            "category_id": int(class_id),
                            "bbox": [x1, y1, x2, y2]
                        })
                    except Exception as e:
                        print(f"解析错误：文件 {label_path} 第{line_num}行 '{line}' -> {str(e)}")
        return {
            "image": image,
            "labels": labels,
            "image_path": str(img_path)
        }

class MSECSPWeightAnalyzer:
    def __init__(self,
                 model_path: Union[str, Path],
                 target_layer: str = 'model.8',
                 img_size: tuple = (3072, 96),
                 device: str = 'cuda:0'):
        """
        改进点：
        1. 增加权重数据记录结构
        2. 集成统计分析组件
        3. 动态权重捕获机制
        """
        # 初始化模型
        self.model = YOLO(str(model_path)).to(device)
        self.model.model.eval()
        self.img_size = img_size  # (width, height)
        self.device = device

        # 数据缓存
        self.weight_records = []
        self.defect_info = []

        # 注册钩子
        self._register_hooks(target_layer)

        # 可视化参数
        self.cmap = cv2.COLORMAP_JET
        self.alpha = 0.5

    def _register_hooks(self, layer_name: str):
        """双钩子注册：捕获权重和特征"""

        def weight_hook(module, input, output):
            # 捕获动态权重 [batch, 3]
            if hasattr(module, 'dms_sp_weights'):
                weights = module.dms_sp_weights.detach().cpu().numpy()
                self.weight_records.extend(weights)

        def feature_hook(module, input, output):
            # 捕获原始特征用于可视化
            self.hook_data = {
                'input': input[0].detach().cpu(),
                'output': output.detach().cpu()
            }

        # 注册到DMS-SP和主模块
        for name, module in self.model.model.named_modules():
            if 'dms_sp' in name.lower():
                module.register_forward_hook(weight_hook)
            if name == layer_name:
                module.register_forward_hook(feature_hook)

    def _extract_defect_metadata(self, targets: List[dict]):
        """提取缺陷元数据"""
        for t in targets:
            bbox = t['bbox']  # 假设格式为 [x1,y1,x2,y2]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            self.defect_info.append({
                'aspect_ratio': max(w / h, h / w),  # 归一化长宽比
                'class_id': t['category_id'],
                'class_name': self._get_class_name(t['category_id'])
            })

    def _get_class_name(self, class_id: int) -> str:
        """类别ID映射（根据实际数据集调整）"""
        class_map = {
            0: '断经',
            1: '双经',
            2: '断纬',
            3: '带纬',
            4: '曲纡'
        }
        return class_map.get(class_id, '未知')

    def _load_dataset(self, root_dir: Path) -> FabricDefectDataset:
        """加载带有标注的织物缺陷数据集"""
        if not (root_dir / 'images').exists():
            raise FileNotFoundError(f"数据集目录缺少images文件夹: {root_dir}")
        if not (root_dir / 'labels').exists():
            print(f"⚠️ 警告: 未找到labels标注文件夹，将仅加载图像数据")

        return FabricDefectDataset(root_dir, img_size=self.img_size)

    def _get_random_sampler(self, dataset: Dataset, sample_size: int) -> RandomSampler:
        """生成随机采样器"""
        if len(dataset) < sample_size:
            print(f"⚠️ 样本量不足: 总样本{len(dataset)} < 需求样本{sample_size}，将使用全部数据")
            sample_size = len(dataset)
        indices = torch.randperm(len(dataset))[:sample_size].tolist()
        return RandomSampler(indices, replacement=False)

    def _custom_collate_fn(self, batch):
        """自定义数据批处理函数，处理不同数量的标注框"""
        images = torch.stack([item['image'] for item in batch])
        labels = [item['labels'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        return {
            'image': images,
            'labels': labels,
            'image_path': image_paths
        }

    def analyze_batch(self,
                      input_dir: Union[str, Path],
                      sample_size: int = 300):
        """
        批量分析流程
        Args:
            input_dir: 包含图像和标注的目录
            sample_size: 随机采样数量
        """
        input_dir = Path(input_dir)

        # 修改后的DataLoader初始化
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            sampler=sampler,
            collate_fn=self._custom_collate_fn  # 添加自定义collate函数
        )

        # 推理循环
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="分析进度"):  # 使用修改后的dataloader
                images = batch['image'].to(self.device)
                self.model(images)  # 触发钩子
                # 处理标签时需要展开批次
                for single_labels in batch['labels']:
                    self._extract_defect_metadata(single_labels)

        # 推理循环
        with torch.no_grad():
            for batch in tqdm(DataLoader(dataset, batch_size=8, sampler=sampler),
                              desc="分析进度"):
                images = batch['image'].to(self.device)
                self.model(images)  # 触发钩子
                self._extract_defect_metadata(batch['labels'])

        # 数据整合
        df = pd.DataFrame({
            'global_h': [w[0] for w in self.weight_records],
            'local_h': [w[1] for w in self.weight_records],
            'vertical': [w[2] for w in self.weight_records],
            ** {k: [d[k] for d in self.defect_info] for k in self.defect_info[0]}
        })

        # 执行分析
        self._perform_analysis(df)

    def _perform_analysis(self, df: pd.DataFrame):
        """执行统计分析"""
        # 1. 类别权重分布
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='class_name', y='global_h', data=df)
        plt.title('全局横向权重分布')
        plt.savefig('global_weight_dist.jpg')

        # 2. 长宽比相关性
        corr_results = {}
        for col in ['global_h', 'local_h', 'vertical']:
            r, p = pearsonr(df['aspect_ratio'], df[col])
            corr_results[col] = (r, p)

        # 3. 可视化相关性
        self._plot_correlation(df)

    def _plot_correlation(self, df: pd.DataFrame):
        """绘制长宽比-权重散点图"""
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        sns.regplot(x='aspect_ratio', y='global_h', data=df,
                    scatter_kws={'alpha': 0.3})
        plt.title(f"Global Horizontal (r={df['global_h'].corr(df['aspect_ratio']):.2f})")

        plt.subplot(132)
        sns.regplot(x='aspect_ratio', y='vertical', data=df,
                    scatter_kws={'alpha': 0.3})
        plt.title(f"Vertical (r={df['vertical'].corr(df['aspect_ratio']):.2f})")

        plt.subplot(133)
        sns.regplot(x='aspect_ratio', y='local_h', data=df,
                    scatter_kws={'alpha': 0.3})
        plt.title(f"Local Horizontal (r={df['local_h'].corr(df['aspect_ratio']):.2f})")

        plt.tight_layout()
        plt.savefig('weight_correlation.jpg')


if __name__ == "__main__":
    # ================== 配置区 ==================
    MODEL_PATH = Path("D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/MAC/weights/best.pt")
    DATA_DIR = Path("D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/VOCdevkit_all_last_split/val//")

    # ================== 执行分析 ==================
    analyzer = MSECSPWeightAnalyzer(
        model_path=MODEL_PATH,
        target_layer='model.8',
        img_size=(3072, 96),
        device='cuda:0'
    )

    analyzer.analyze_batch(
        input_dir=DATA_DIR,
        sample_size=300
    )