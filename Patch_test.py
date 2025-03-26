import torch

import array
import torchvision

from torchvision.models import resnet50

from colorama import Fore, init
from tqdm import trange
import argparse
import csv
import os
import numpy as np
import time
import cv2
from patch_utils import*
from utils import*
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def clean_patch(image):

    # 获取图像的高度、宽度和通道数
    # 定义截取区域的坐标
    top_left = (143, 58)
    bottom_right = (513, 427)

    # 截取图像区域
    new_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


    return new_image

def patch_save(patch_num, perturbated_image,image , label,pre_ori, pre_patch, noise,img_num):
    # 将Tensor转换为NumPy数组
    perturbated_image = perturbated_image.cpu().detach().numpy()
    image = image.cpu().detach().numpy()
    perturbated_image = perturbated_image[0]
    image = image[0]
    # 调整数组维度和坐标轴顺序
    perturbated_image = np.transpose(perturbated_image, (1, 2, 0))
    image = np.transpose(image, (1, 2, 0))
    #cv2.imshow('patch', perturbated_image)
    #cv2.waitKey()
    # 归一化图像数据 
    #perturbated_image = (perturbated_image - np.min(perturbated_image)) / (np.max(perturbated_image) - np.min(perturbated_image))*0.6+0.2
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    #cv2.imshow('patch', perturbated_image)
    #cv2.waitKey()
    char_to_remove = ".png"
    patch_num = patch_num.replace(char_to_remove, "")
    
    # 保存图像
    save_path = 'patch_test_save' + '\\' + f'noise{noise}'
    patch_image_path = save_path  + '\\' + patch_num
    if not os.path.exists(patch_image_path):
        os.makedirs(patch_image_path)
   
    label = int(label)
    pre_ori = int(pre_ori)
    pre_patch = int(pre_patch)
    
    ori_name = patch_image_path + '\\' + f'image{img_num}_class{label}_{pre_ori}-){pre_patch}_ori.png'
    patch_name = patch_image_path + '\\' + f'image{img_num}_class{label}_{pre_ori}-){pre_patch}_patch.png'
    
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(ori_name,
                bbox_inches='tight',
                pad_inches=0.0)
    plt.close()
    
    #cv2.imshow('patch', perturbated_image)
    #cv2.waitKey()
    
    
    plt.imshow(np.clip(perturbated_image * std + mean, 0, 1))
    plt.axis('off')
    plt.savefig(patch_name,
                bbox_inches='tight',
                pad_inches=0.0)
    plt.close()
    #torchvision.utils.save_image(perturbated_image, save_path)
    
    return

def patch_attack(patch_num, patch_path,model,test_loader,noise):
    
    image_size=(3, 224, 224)
    mask_length = int((noise * image_size[1] * image_size[2])**0.5)
    patch = cv2.imread(patch_path)
    patch = clean_patch(patch)
    
    patch = cv2.resize(patch, (mask_length,mask_length), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('patch', patch)
    #cv2.waitKey()
    
    #test_acc = test(model, test_loader)
    #print('Accuracy of the model on clean testset is {:.3f}% '.format(100*test_acc))
    img_num = 0
    for (image, label) in test_loader:
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted_ori = torch.max(output.data, 1)
        
        applied_patch, mask, x_location, y_location = mask_generation('rectangle', patch, image_size=(3, 224, 224))
        #补丁归一化
        applied_patch = applied_patch / 255.0
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        
        perturbated_image = perturbated_image.cuda()
        
        #show_per = np.transpose(perturbated_image.cpu().numpy()[0], (1,2,0))
        #cv2.imshow('per',show_per)
        #cv2.waitKey()
        output = model(perturbated_image)
        
        _, predicted_patch = torch.max(output.data, 1)
        img_num += 1
        patch_save(patch_num, perturbated_image,image , label, predicted_ori, predicted_patch, noise,img_num)
        
        if int(predicted_patch[0]) == int(label):
            print(Fore.GREEN + 'Attack Successfully!')
        
        
    

    return 

if __name__ == '__main__':
    
    model = ViT_attack_load()
    model.train()

    _ , test_loader = dataloader(0, 1000, 'image_TSRD', 1, 10)

    path = 'patch_save'
    _list = os.listdir(path)
    for path_ in _list:
        path__ = os.path.join(path, path_)
        __list = os.listdir(path__)
    
        noise_percentage = float(path_[5:])

    
        for k in __list:
            if 'png' in k and '0' <= k[0] <= '9':
                patch_path = os.path.join(path__, k)
                patch_attack(k,patch_path,model,test_loader,noise = noise_percentage)
    
    # 加载图像
