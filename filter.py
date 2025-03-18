# import os
# import shutil
#
# def find_txt_with_zero(label_train_dir):
#     txt_with_zero = []
#     for txt_file in os.listdir(label_train_dir):
#         if txt_file.endswith(".txt"):
#             txt_path = os.path.join(label_train_dir, txt_file)
#             with open(txt_path, "r") as f:
#                 lines = f.readlines()
#             for line in lines:
#                 parts = line.strip().split()
#                 if len(parts) > 0 and parts[0] == "0":
#                     txt_name = os.path.splitext(txt_file)[0]
#                     txt_with_zero.append(txt_name)
#                     break
#     return txt_with_zero
#
# def copy_images(label_train_dir, image_train_dir, filter_list, output_dir):
#     # 创建 images 和 labels 子文件夹
#     output_images_dir = os.path.join(output_dir, "images")
#     output_labels_dir = os.path.join(output_dir, "labels")
#     if not os.path.exists(output_images_dir):
#         os.makedirs(output_images_dir)
#     if not os.path.exists(output_labels_dir):
#         os.makedirs(output_labels_dir)
#
#     for name in filter_list:
#         image_png_path = os.path.join(image_train_dir, name + ".png")
#         image_jpg_path = os.path.join(image_train_dir, name + ".jpg")
#         label_path = os.path.join(label_train_dir, name + ".txt")
#
#         output_image_png_path = os.path.join(output_images_dir, name + ".png")
#         output_image_jpg_path = os.path.join(output_images_dir, name + ".jpg")
#         output_label_path = os.path.join(output_labels_dir, name + ".txt")
#
#         if os.path.exists(image_png_path):
#             shutil.copy(image_png_path, output_image_png_path)
#             shutil.copy(label_path, output_label_path)
#             print(f"Copied {name}.png to {output_images_dir} and {name}.txt to {output_labels_dir}")
#         elif os.path.exists(image_jpg_path):
#             shutil.copy(image_jpg_path, output_image_jpg_path)
#             shutil.copy(label_path, output_label_path)
#             print(f"Copied {name}.jpg to {output_images_dir} and {name}.txt to {output_labels_dir}")
#         else:
#             print(f"Image for {name} not found in {image_train_dir}")
#
#     print(f"Total files processed: {len(filter_list)}")
#
# # 参数定义
# title = "val"
# root_path = r"F:\BaiduNetdiskDownload\4.2k HRW yolo dataset\4.2k_HRW_yolo_dataset"
# label_train_dir = os.path.join(root_path, "labels", title)
# image_train_dir = os.path.join(root_path, "images", title)
# output_dir = fr"D:\pythonProject\head_examine\head_examine\{title}"
#
# # 创建 output_dir（如果不存在）
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 获取包含类别 0 的 txt 文件列表
# filter_list = find_txt_with_zero(label_train_dir)
#
# # 执行复制
# copy_images(label_train_dir, image_train_dir, filter_list, output_dir)

import os
import shutil

# 查找包含类别 0 的 txt 文件
def find_txt_with_zero(label_train_dir):
    txt_with_zero = []
    for txt_file in os.listdir(label_train_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(label_train_dir, txt_file)
            with open(txt_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0] == "0":
                    txt_name = os.path.splitext(txt_file)[0]
                    txt_with_zero.append(txt_name)
                    break
    return txt_with_zero

# 查找标签全为类别 0 的 txt 文件
def find_txt_all_zero(label_train_dir):
    txt_all_zero = []
    for txt_file in os.listdir(label_train_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(label_train_dir, txt_file)
            with open(txt_path, "r") as f:
                lines = f.readlines()
            if not lines:  # 空文件跳过
                continue
            all_zero = True
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0 and parts[0] != "0":  # 如果有非 0 类别
                    all_zero = False
                    break
            if all_zero:
                txt_name = os.path.splitext(txt_file)[0]
                txt_all_zero.append(txt_name)
    return txt_all_zero

# 复制图像和标签，区分是否全为类别 0
def copy_images(label_train_dir, image_train_dir, filter_list, output_dir):
    # 创建 images 和 labels 子文件夹
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for name in filter_list:
        image_png_path = os.path.join(image_train_dir, name + ".png")
        image_jpg_path = os.path.join(image_train_dir, name + ".jpg")
        label_path = os.path.join(label_train_dir, name + ".txt")

        output_image_png_path = os.path.join(output_images_dir, name + ".png")
        output_image_jpg_path = os.path.join(output_images_dir, name + ".jpg")
        output_label_path = os.path.join(output_labels_dir, name + ".txt")

        if os.path.exists(image_png_path):
            shutil.copy(image_png_path, output_image_png_path)
            shutil.copy(label_path, output_label_path)
            print(f"Copied {name}.png to {output_images_dir} and {name}.txt to {output_labels_dir}")
        elif os.path.exists(image_jpg_path):
            shutil.copy(image_jpg_path, output_image_jpg_path)
            shutil.copy(label_path, output_label_path)
            print(f"Copied {name}.jpg to {output_images_dir} and {name}.txt to {output_labels_dir}")
        else:
            print(f"Image for {name} not found in {image_train_dir}")

    print(f"Total files processed: {len(filter_list)}")

# 参数定义
title = "val"
root_path = r"F:\BaiduNetdiskDownload\4.2k HRW yolo dataset\4.2k_HRW_yolo_dataset"
label_train_dir = os.path.join(root_path, "labels", title)
image_train_dir = os.path.join(root_path, "images", title)
output_dir = fr"D:\pythonProject\rec_head\datasets\{title}"

# 创建 output_dir（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取包含类别 0 的 txt 文件列表
# filter_list = find_txt_with_zero(label_train_dir)
filter_list = find_txt_all_zero(label_train_dir)
# 执行复制
copy_images(label_train_dir, image_train_dir, filter_list, output_dir)