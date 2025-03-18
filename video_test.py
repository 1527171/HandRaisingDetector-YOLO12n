import cv2
from ultralytics import YOLO
from tqdm import tqdm

# 1. 加载保存的模型
model = YOLO(r".\model\modele200b16.pt")

# 2. 打开输入视频
video_name = "20250318_151446.mp4"


input_video_path = fr".\video\{video_name}"
cap = cv2.VideoCapture(input_video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("错误：无法打开视频。")
    exit()

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

# 3. 定义输出视频路径和编码器
output_video_path = fr".\video_output\{video_name}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 4. 创建进度条
progress_bar = tqdm(total=total_frames, desc="处理视频", unit="帧")

# 5. 处理视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频处理完成。")
        break

    # 使用模型进行推理，禁用详细输出
    results = model(frame, conf=0.3, verbose=False)

    # 绘制检测结果（举手框）
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 假设 0 表示 'hand-raising'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 绘制绿色边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加置信度标签，动态调整位置
                text = f'hand-raising: {conf:.2f}'
                font_scale = 0.9
                font_thickness = 2
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_size[1] + 10
                if text_x + text_size[0] > frame.shape[1]:
                    text_x = x2 - text_size[0] - 10
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 0), font_thickness)

    # 写入处理后的帧到输出视频
    out.write(frame)

    # 更新进度条
    progress_bar.update(1)

# 6. 释放资源
progress_bar.close()
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"输出视频已保存为 {output_video_path}")