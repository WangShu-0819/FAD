import os
from ultralytics import YOLO
import warnings



warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")

# 多线程添加代码
if __name__ == '__main__':

    # 训练模型
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()

    # 加载训练好的模型
    model = YOLO('D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/MAC/weights/last.pt')

    # 验证配置
    results = model.val(
        data='data.yaml',  # 数据集配置文件
        split='val',  # 验证集划分
        batch=1,  # 根据显存调整
        imgsz=[48, 1536],  # 训练使用的图像尺寸
        conf=0.01,  # 与训练配置保持一致
        iou=0.3,  # NMS IoU阈值
        device='0',  # 使用GPU 0
        plots=True,  # 生成PR曲线和混淆矩阵
        save_json=True,  # 保存JSON格式结果
        # save_hybrid=True,  # 保存混合精度结果
        save=True,  # 保存预测结果
        save_txt=True,  # 保存预测框的文本文件
        save_conf=True,  # 保存预测置信度
        visualize=True,  # 可视化预测结果
    )

    # 假设 results 是你的评估结果对象
    # 提取需要的指标
    precision = results.box.p
    recall = results.box.r
    map50 = results.box.map50
    map25 = results.box.map25
    map50_95 = results.box.map
    f1_score = results.box.f1

