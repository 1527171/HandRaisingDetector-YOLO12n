from ultralytics import YOLO
import cv2

# 1. 加载保存的最佳模型
model = YOLO('modele100b16.pt')

# 2. 加载图片并统一调整大小
pic = '4.png'
image_path = fr"test/{pic}"  # 替换为实际图片路径
img = cv2.imread(image_path)

# 调整图片大小为 640x640
img_resized = cv2.resize(img, (640, 640))

# 3. 使用调整后的图片进行推理
results = model(img_resized, conf=0.4)  # 设置置信度阈值

# 4. 处理推理结果并可视化
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])  # 获取类别 ID
        if cls_id == 0:  # 假设 'hand-raising' 类别 ID 为 0
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
            conf = float(box.conf[0])  # 获取置信度
            text = f'hand-raising {conf:.2f}'  # 文字内容

            # 绘制边界框
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

            # 动态调整文字位置
            font_scale = 0.9
            font_thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10

            # 如果文字超出顶部，移到框内
            if text_y < 0:
                text_y = y1 + text_size[1] + 10  # 移到框内下方

            # 如果文字超出右侧，移到框内
            if text_x + text_size[0] > img_resized.shape[1]:
                text_x = x2 - text_size[0] - 10  # 移到框内右侧

            # 绘制文字
            cv2.putText(img_resized, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), font_thickness)

# 5. 显示推理结果（支持伸缩）
window_name = 'Detection Result'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
cv2.imshow(window_name, img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 可选：保存推理结果
cv2.imwrite(f'test_out/{pic}', img_resized)