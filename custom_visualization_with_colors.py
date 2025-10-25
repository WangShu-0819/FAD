import os
import cv2
import numpy as np


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


# 定义函数来绘制边界框
def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2, class_id = map(int, box[:5])
        color = colors(class_id, bgr=True)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


# 分别定义图片路径、标签路径和保存路径
image_path = 'D:/study/Textile_defects/ultraytics/ultralytics-main/datasets/VOCdevkit_all_last_split/val/images'
label_path = 'D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/val_yolov8_2/labels'
save_path = 'D:/study/Textile_defects/ultralytics-yolov8/ultralytics-main/runs/detect/yolov8_visualization'

# 创建保存路径
os.makedirs(save_path, exist_ok=True)

# 遍历验证集图片
for img_name in os.listdir(image_path):
    img_path = os.path.join(image_path, img_name)
    txt_name = img_name.replace('.jpg', '.txt')
    txt_path = os.path.join(label_path, txt_name)

    # 读取图片
    image = cv2.imread(img_path)

    # 读取预测结果
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            boxes = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                img_height, img_width = image.shape[:2]
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                boxes.append([x1, y1, x2, y2, class_id])

        # 绘制边界框
        image = draw_boxes(image, boxes)

    # 保存图片
    save_img_path = os.path.join(save_path, img_name)
    cv2.imwrite(save_img_path, image)
