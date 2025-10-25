from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/yolov8/weights/last.pt"
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')  # 从YAML建立一个新模型

    test_path = r"D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/VOCdevkit_all_last_split/val/images"
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.01, imgsz=[1536, 48], save_conf=True, save_txt=True, name='output')  # predict on an image