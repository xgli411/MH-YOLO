import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/runs1/train/helmet-detection/weights/best.pt')# 加载自己训练的模型
    metrics=model.val(data='public1/your.yaml',#
                split='test',
                # save_json=True, # if you need to cal coco metrice
                project='/runs/runs1/val',
                name='helmet-detection',
                )# 训练模型
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75ai
    metrics.box.maps   # a list contains map50-95 of each category
