import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning, module="torch.autograd")

# 多线程添加代码
if __name__ == '__main__':
    # 加载一个模型 ,workers=8
    model = YOLO('ultralytics/cfg/models/v8/yolov8-ours.yaml')  # 从YAML建立一个新模型
    # model.load('yolov8n.pt')
    # 替换损失函数
    # model.model.loss = MultiHeadDetectionLoss(model.model)
    # model.iou_thres = 0.0001  # 设置 IoU 阈值为 0.25
    # confidence_threshold = 0.25.

    # 训练模型
    import torch, gc
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 强制允许重复的OpenMP运行时

    gc.collect()
    torch.cuda.empty_cache()

    results = model.train(data='data.yaml', epochs=200, imgsz=[96, 3072], rect=True, device=0, workers=8, batch=8,
                          cache=True)
