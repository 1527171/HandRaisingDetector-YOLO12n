# HandRaisingDetector-YOLO12n

**HandRaisingDetector-YOLO12n** 是一个基于 YOLO12n 的目标检测系统，旨在自动检测视频中学生举手的行为。该项目利用深度学习技术实时识别举手状态，可用于教育场景，帮助教师监控课堂互动并提升教学效率。

---

## 数据集来源
https://github.com/Whiffe/SCB-dataset
   [**Student Classroom Behavior Detection based on Improved YOLOv7**]

## filter.py

将下载好的数据集筛选出为举手动作的数据并保存

## train.py

yolo推理训练，model文件夹为训练好的模型，e100b16指epochs为100，batch为16.

训练效果：


                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:03<00:00,  4.52it/s]
                   all        560       1737      0.745       0.67      0.762      0.513

## test.py

对test图片进行测试

![1](https://github.com/1527171/HandRaisingDetector-YOLO12n/blob/master/test_out/4.png)


## video_test.py

将视频中举手同学进行标注

## 克隆仓库
```bash
git clone https://github.com/your-username/HandRaisingDetector-YOLO12n.git
cd HandRaisingDetector-YOLO12n

