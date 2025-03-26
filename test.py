import os
import matplotlib.pyplot as plt
import numpy as np

# 图片文件夹路径
patch_save_dir = "patch_save"
result_dir = "result/processed"

# 创建保存结果的文件夹
os.makedirs(result_dir, exist_ok=True)

#储存图片信息的列表
noise_list = [0.02, 0.04, 0.06, 0.08, 0.10]

image_name_info = np.empty((5, 20, 9, 10), dtype=np.dtype('U10'))
noise_num = 0
# 遍历文件夹，读取图片并分类存储
for noise_folder in os.listdir(patch_save_dir):
    noise = noise_folder.replace("noise", "")
    noise_num = noise_list.index(float(noise))
    noise_folder_path = os.path.join(patch_save_dir, noise_folder)
    if os.path.isdir(noise_folder_path):
        noise_info = {}
        for ori_patch_folder in os.listdir(noise_folder_path):
            ori_patch_folder_path = os.path.join(noise_folder_path, ori_patch_folder)
            if os.path.isdir(ori_patch_folder_path):

                files = os.listdir(ori_patch_folder_path)
                for file in os.listdir(ori_patch_folder_path):
                    file = file.replace(".png", "")
                    file = file.split("_")
                    img_all_num = int(file[0].replace("num", ""))
                    epoch = int(file[4].replace("epoch", ""))
                    img_num = (img_all_num/100)%9
                    img_num =int(img_num)
                    if file[1] == "ori":
                        image_name_info[noise_num,epoch,img_num,0:5] = file
                        #print(image_name_info[noise_num,epoch_num,img_num,0:5])
                    elif file[1] == 'patch':
                        image_name_info[noise_num,epoch,img_num,5:10] = file
                        #print(image_name_info[noise_num,epoch_num,img_num])
                    else:
                        print("error")
    print(image_name_info)                    
                    

                    
             
                        
print(image_name_info)  


# 处理图片信息,画图并保存




for noise_num in range(5):
    epoch_acc_list = []
    for epoch_num in range(20):
        
        epoch_acc = 0.0
        for img_num in range(9):
            img_1 = image_name_info[noise_num,epoch_num,img_num,0:5]
            img_2 = image_name_info[noise_num,epoch_num,img_num,5:10]
            
            label = img_1[2].replace("label", "")
            pre = img_1[3].replace("pre", "")
            attack_pre = img_2[3].replace("pre", "")
            if attack_pre == '0':
                epoch_acc+=1
        epoch_acc_rate = epoch_acc/9.0
        epoch_acc_list.append(epoch_acc_rate)
    
    plt.plot(range(1,21), epoch_acc_list,label = f"noise{noise_list[noise_num]}")
    plt.legend(loc=0,ncol=2)
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.title(f"noise")
    plt.savefig(f"{result_dir}/noise{noise_list[noise_num]}.png")

    
        