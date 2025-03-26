import os
import matplotlib.pyplot as plt
import numpy as np

# 图片文件夹路径
patch_save_dir = "patch_save"
result_dir = "result/processed"

# 创建保存结果的文件夹
os.makedirs(result_dir, exist_ok=True)

# 储存图片信息的字典
image_name_info = {}

noise_num = 0
# 遍历文件夹，读取图片并分类存储
for noise_folder in os.listdir(patch_save_dir):
    
    noise_folder_path = os.path.join(patch_save_dir, noise_folder)
    if os.path.isdir(noise_folder_path):
        noise_info = {}
        for ori_patch_folder in os.listdir(noise_folder_path):
            ori_patch_folder_path = os.path.join(noise_folder_path, ori_patch_folder)
            if os.path.isdir(ori_patch_folder_path):
                epoch_num = 0
                img_num = 0
                for file in os.listdir(ori_patch_folder_path):
                    file = file.replace(".png", "")
                    file = file.split("_")
                    if noise_num not in noise_info:
                        noise_info[noise_num] = {}
                    if epoch_num not in noise_info[noise_num]:
                        noise_info[noise_num][epoch_num] = {}
                    noise_info[noise_num][epoch_num][img_num] = {
                        'file': file,
                        'label': file[2].replace("label", ""),
                        'pre': file[3].replace("pre", "")
                    }
                    img_num += 1
                    if img_num == 18:
                        img_num = 0
                        epoch_num += 1
        image_name_info.update(noise_info)
    noise_num += 1

print(image_name_info)


# 处理图片信息，画图并保存

noise_lise = [0.02, 0.04, 0.06, 0.08, 0.1]

for noise_num in range(5):
    epoch_acc_list = []
    for epoch_num in range(20):
        epoch_acc = 0.0
        for img_num in range(9):
            img_info_1 = image_name_info.get(noise_num, {}).get(epoch_num, {}).get(img_num)
            img_info_2 = image_name_info.get(noise_num, {}).get(epoch_num, {}).get(img_num + 1)
            if img_info_1 and img_info_2:
                label = img_info_1['label']
                pre = img_info_1['pre']
                attack_pre = img_info_2['pre']
                if attack_pre == '0':
                    epoch_acc += 1
        epoch_acc_rate = epoch_acc / 9.0
        epoch_acc_list.append(epoch_acc_rate)

    plt.plot(range(1, 21), epoch_acc_list, label=f"noise{noise_lise[noise_num]}")
    plt.legend(loc=0, ncol=2)
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.title(f"noise")
    plt.savefig(f"{result_dir}/noise{noise_lise[noise_num]}.png")