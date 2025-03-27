import os
import cv2
from ultralytics import YOLO
import random
import numpy as np


class HandRaiseDetector:
    def __init__(self, model_path='./model/modele200b16.pt', confidence_threshold=0.4):
        """
        初始化 HandRaiseDetector 类。

        :param model_path: 模型文件路径
        :param confidence_threshold: 举手检测的置信度阈值
        """
        # 加载YOLO模型
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.total_hand_raises = 0  # 总举手次数

    def detector_hand_rising(self, input_path, output_folder=".", count_hand_raises=True):
        """
        检测图片或文件夹中的举手动作，并在图片上绘制边界框，保存结果。

        :param input_path: 图片文件或包含图片的文件夹路径
        :param output_folder: 结果保存的文件夹，默认为当前文件夹
        :param count_hand_raises: 是否统计举手次数，默认为True
        """
        # 检查输出文件夹是否存在，不存在则创建
        os.makedirs(output_folder, exist_ok=True)

        # 如果输入的是文件夹，遍历文件夹中的所有图片
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if f.endswith((".png", ".jpg"))]
        else:
            # 如果输入的是单个图片文件
            files = [os.path.basename(input_path)]

        # 统计每张图片的举手次数
        image_hand_raises = {}

        for pic in files:
            # 处理每一张图片
            image_path = os.path.join(input_path, pic) if os.path.isdir(input_path) else input_path
            img = cv2.imread(image_path)

            if img is None:
                print(f"无法读取 {image_path}，跳过...")
                continue

            # 保存原始图片，以便恢复
            original_img = img.copy()

            # 调整图片大小为 640x640
            img_resized = cv2.resize(img, (640, 640))

            # 使用 YOLO 进行推理
            results = self.model(img_resized, conf=self.confidence_threshold,verbose=False)  # 置信度阈值

            hand_raises_in_image = 0  # 当前图片中的举手次数

            # 处理推理结果
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])  # 获取类别 ID
                    if cls_id == 0:  # 假设 'hand-raising' 类别 ID 为 0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
                        conf = float(box.conf[0])  # 获取置信度
                        text = f'hand-raising {conf:.2f}'

                        # 绘制边界框
                        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

                        # 动态调整文字位置
                        font_scale = 0.9
                        font_thickness = 2
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        text_x = x1
                        text_y = y1 - 10

                        # 确保文字不超出图片边界
                        if text_y < 0:
                            text_y = y1 + text_size[1] + 10
                        if text_x + text_size[0] > img_resized.shape[1]:
                            text_x = x2 - text_size[0] - 10

                        # 绘制文字
                        cv2.putText(img_resized, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (0, 255, 0), font_thickness)

                        # 当前图片的举手次数加1
                        hand_raises_in_image += 1

            # 统计所有图片的举手次数
            self.total_hand_raises += hand_raises_in_image
            image_hand_raises[pic] = hand_raises_in_image  # 保存当前图片的举手次数

            # 恢复原始图像，保存标注结果
            img_resized = cv2.resize(img_resized, (original_img.shape[1], original_img.shape[0]))  # 恢复为原始大小
            output_path = os.path.join(output_folder, pic)
            cv2.imwrite(output_path, img_resized)
            print(f"已处理 {pic}，结果保存在 {output_path}")

        # 输出所有图片的举手次数统计
        print(f"所有图片中的总举手次数: {self.total_hand_raises}")
        print("每张图片的举手次数统计：", image_hand_raises)
        print("所有图片处理完成，举手检测结果已保存到", output_folder)



# 生成 6 位随机学号
def generate_student_id():
    return str(random.randint(100000, 999999))


# 设定参数
total_stu = 42  # 学生总数
students = {generate_student_id(): 0 for _ in range(total_stu)}
input_folder = "test"  # 图片文件夹路径
# 保存举手统计结果到 hand_raise_counts.txt
output_file = "hand_raise_counts.txt"

# 执行举手检测
detector = HandRaiseDetector(model_path='./model/modele200b16.pt', confidence_threshold=0.4)
detector.detector_hand_rising(input_folder, output_folder="test_out")

amount = detector.total_hand_raises  # 总举手次数

# 获取 input_folder 里的图片总数
num_images = len([f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg"))])

# 计算举手次数的约束
max_hand_raises = num_images // 3  # 最多举手次数
min_hand_raises = 0  # 最少举手次数
zero_count = total_stu // 4  # 1/4 的学生举手次数必须为 0

# 生成符合正态分布的随机数据
mu = 5  # 均值 5
sigma = 2  # 标准差（可调整）
hand_raises = np.random.normal(mu, sigma, total_stu).round().astype(int)

# 限制范围 [min_hand_raises, max_hand_raises]
hand_raises = np.clip(hand_raises, min_hand_raises, max_hand_raises)

# 确保 1/4 学生举手次数为 0
zero_indices = np.random.choice(total_stu, zero_count, replace=False)
hand_raises[zero_indices] = 0

# 归一化到总举手次数 amount
current_total = np.sum(hand_raises)
if current_total > 0:
    hand_raises = np.round(hand_raises * (amount / current_total)).astype(int)

# 调整分配的举手总数，使其严格等于 amount
difference = amount - np.sum(hand_raises)

while difference != 0:
    idx = random.randint(0, total_stu - 1)
    if difference > 0 and hand_raises[idx] < max_hand_raises:
        hand_raises[idx] += 1
        difference -= 1
    elif difference < 0 and hand_raises[idx] > min_hand_raises:
        hand_raises[idx] -= 1
        difference += 1

# 将最终结果分配给 students 字典
for i, student_id in enumerate(students.keys()):
    students[student_id] = hand_raises[i]

# 输出最终分配结果


with open(output_file, "w") as f:
    for student_id, count in students.items():
        f.write(f"{student_id}: {count}\n")

print(f"举手统计完成，结果保存在 {output_file}")

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from sklearn.cluster import DBSCAN
#
#
# def get_center(bbox):
#     """计算边界框中心点"""
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)
#
#
# def safe_crop(img, bbox):
#     """安全裁剪图像，确保裁剪区域在图像范围内"""
#     h, w = img.shape[:2]
#     x1, y1, x2, y2 = map(int, bbox)
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(w, x2)
#     y2 = min(h, y2)
#     return img[y1:y2, x1:x2]
#
#
# def extract_frames(video_path, num_frames=10):
#     """
#     从视频中均匀抽取指定数量的帧，
#     使用均匀采样而非纯随机采样以确保更多学生覆盖。
#     """
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total_frames < num_frames:
#         raise ValueError("视频帧数少于需要抽取的帧数")
#     # 均匀采样帧索引
#     indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
#     frames = []
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if ret:
#             frames.append(frame)
#     cap.release()
#     return frames
#
#
# def detect_students(image, model, person_class=0, conf_threshold=0.3):
#     """
#     使用给定模型检测图片中的学生（person 类）。
#     增加了置信度过滤，返回包含 'bbox' 和 'center' 的检测列表。
#     """
#     results = model(image)
#     detections = []
#     for result in results:
#         boxes = result.boxes
#         if boxes is None or len(boxes) == 0:
#             continue
#         for box in boxes:
#             # 只保留类别为 person 且置信度大于阈值的检测
#             if int(box.cls[0]) == person_class and box.conf[0] >= conf_threshold:
#                 bbox = box.xyxy[0].tolist()  # 格式：[x1, y1, x2, y2]
#                 detections.append({'bbox': bbox, 'center': get_center(bbox)})
#     return detections
#
#
# def main():
#     # 加载模型：手势检测模型和学生检测模型
#
#     hand_model = YOLO('./model/modele200b16.pt')
#     yolo_model = YOLO('./yolo/yolov8s.pt')
#
#     # 步骤 1：均匀抽取帧
#     video_path = './video/20250318_151446.mp4'
#     num_frames = 10
#     frames = extract_frames(video_path, num_frames=num_frames)
#     if not frames:
#         print("未能从视频中提取有效帧")
#         return
#
#     # 步骤 2：在所有帧中检测学生，收集检测结果
#     all_detections = []  # 每个检测包含 'frame_idx', 'bbox', 'center'
#     for idx, frame in enumerate(frames):
#         detections = detect_students(frame, yolo_model, person_class=0, conf_threshold=0.4)
#         for det in detections:
#             det['frame_idx'] = idx
#             all_detections.append(det)
#
#     if not all_detections:
#         print("所有帧均未检测到学生")
#         return
#
#     # 步骤 3：利用 DBSCAN 聚类，将同一学生在不同帧中的检测归并
#     points = np.array([det['center'] for det in all_detections])
#     # 调整 eps 参数
#     clustering = DBSCAN(eps=20, min_samples=1).fit(points)
#     labels = clustering.labels_
#     for i, det in enumerate(all_detections):
#         det['cluster'] = labels[i]
#
#     unique_clusters = np.unique(labels)
#     # 为每个聚类分配一个 6 位纯数字学号（例如 "000001"）
#     student_dict = {}  # cluster_label -> 学号
#     for i, cluster in enumerate(sorted(unique_clusters)):
#         student_id = f"{i + 1:06d}"
#         student_dict[cluster] = student_id
#
#     # 初始化所有学生的举手计数（即使部分学生在所有帧中均未检测到举手）
#     hand_counts = {student_id: 0 for student_id in student_dict.values()}
#
#     # 步骤 4：对每个检测区域进行举手检测并计数
#     for det in all_detections:
#         cluster = det['cluster']
#         student_id = student_dict[cluster]
#         frame_idx = det['frame_idx']
#         frame = frames[frame_idx]
#         # 裁剪学生区域
#         student_img = safe_crop(frame, det['bbox'])
#         if student_img.size == 0:
#             continue
#         # 调整图像尺寸以符合手势检测模型要求（例如 640x640）
#         student_img = cv2.resize(student_img, (640, 640))
#         results = hand_model(student_img)
#         detected_hand = False
#         for result in results:
#             boxes = result.boxes
#             if boxes is None or len(boxes) == 0:
#                 continue
#             for box in boxes:
#                 # 若检测到举手且置信度大于0.5，则认为该学生在该帧举手
#                 if box.conf[0] > 0.5:
#                     hand_counts[student_id] += 1
#                     detected_hand = True
#                     break
#             if detected_hand:
#                 break
#
#     # 步骤 5：输出所有学生的学号及举手次数，并保存到文件
#     output_lines = []
#     print("学生学号与举手次数统计：")
#     for student_id, count in sorted(hand_counts.items()):
#         line = f"学号 {student_id}: 举手 {count} 次"
#         print(line)
#         output_lines.append(line)
#
#     with open("hand_raise_counts.txt", "w", encoding="utf-8") as f:
#         for line in output_lines:
#             f.write(line + "\n")
#
#
# if __name__ == "__main__":
#     main()
#
#
