from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import cv2

# 1. 加载模型配置（使用预训练权重）
model = YOLO('yolo12n.pt')  # 使用预训练权重以提高精度和收敛速度

# 2. 定义数据集配置文件
data_yaml = """
train: ./datasets/train/images
val: ./datasets/val/images
nc: 1
names: ['hand-raising']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml)

if __name__ == '__main__':
    freeze_support()
    # 3. 训练模型并显示训练过程
    model.train(
        data='data.yaml',
        epochs=200,           # 增加训练轮次以充分学习
        batch=16,             # 批次大小，根据显存调整
        imgsz=640,           # 输入图像尺寸
        workers=4,           # 数据加载线程数
        verbose=True,        # 显示详细训练过程
        device='cuda',       # 使用 GPU 加速
        # 数据增强参数
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # HSV 颜色增强
        degrees=10.0,        # 旋转角度
        translate=0.2,       # 平移
        scale=0.5,           # 缩放
        shear=15.0,          # 剪切
        flipud=0.5,          # 上下翻转概率
        fliplr=0.5,          # 左右翻转概率
        mosaic=1.0,          # Mosaic 数据增强
        # 学习率和优化参数
        lr0=0.001,           # 初始学习率
        lrf=0.01,            # 最终学习率因子
        momentum=0.937,      # 动量
        weight_decay=0.0005, # 权重衰减
        warmup_epochs=3.0,   # 预热轮次
        warmup_momentum=0.8, # 预热动量
        warmup_bias_lr=0.1,  # 预热偏置学习率
        # 损失函数权重
        box=7.5, cls=0.5, dfl=1.5,
        save_dir='runs/detect/train'  # 训练结果保存路径
    )

    # 4. 推理并只框出举手的人
    image_path = r"F:\BaiduNetdiskDownload\4.2k HRW yolo dataset\4.2k_HRW_yolo_dataset\images\0001003.png"
    results = model(image_path, conf=0.5,iou=0.5)  # 设置置信度阈值为 0.5

    # 加载原始图片
    img = cv2.imread(image_path)

    # 过滤只显示 'hand-raising' 类 (类别ID为0)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # 类别ID
            if cls_id == 0:  # 'hand-raising' 对应的ID是 0
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
                conf = float(box.conf[0])  # 置信度
                # 在图片上绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
                cv2.putText(img, f'hand-raising {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Hand-raising Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存推理结果（可选）
    cv2.imwrite('output_hand_raising.jpg', img)

    # 5. 保存最佳模型
    model.save('modele200b16.pt')  # 保存训练后的最佳模型

