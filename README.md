# HandRaisingDetector-YOLO12n

**HandRaisingDetector-YOLO12n** 是一个基于 YOLO12n 的目标检测系统，旨在自动检测视频中学生举手的行为。该项目利用深度学习技术实时识别举手状态，可用于教育场景，帮助教师监控课堂互动并提升教学效率。

---

## 数据集来源
https://github.com/Whiffe/SCB-dataset
   [**Student Classroom Behavior Detection based on Improved YOLOv7**]

## filter.py

将下载好的数据集筛选出为举手动作的数据并保存

## train.py

yolo推理训练

## test.py

对test图片进行测试

## video_test.py

将视频中举手同学进行标注
## 安装步骤

### 克隆仓库
```bash
git clone https://github.com/your-username/HandRaisingDetector-YOLOv8.git
cd HandRaisingDetector-YOLOv8

