# Adversarial Patch: utils
# Utils in need to generate the patch and test on the dataset.
# Created by Junbo Zhao 2020/3/19

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import ResNet50_Weights,resnet50,vision_transformer


import time
# 时间装饰器
def Timer_Decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_time = round(execution_time, 2)
        
        print(f"{func.__name__}消耗时间{execution_time}s")
        return result
    return wrapper


# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
@Timer_Decorator
def dataloader(train_size, test_size, data_dir, batch_size, num_workers,lin = 224):
    total_num = train_size + test_size
    # 图像预处理，依次进行随机裁剪、水平翻转、归一化
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(lin),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(size=(lin, lin)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]# 训练集长度[0,train_size]
    test_index = index[train_size: (train_size + test_size)]# 测试集长度[train_size,train_size+test_size]

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=test_transforms)

    
    # 按照索引划分数据集，sampler是用来指定数据集的采样策略,SubsetRandomSampler是指定采样策略为随机采样,并且按照给定的索引采样
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              sampler=SubsetRandomSampler(train_index), 
                              num_workers=num_workers, 
                              pin_memory=True, 
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             sampler=SubsetRandomSampler(test_index), 
                             num_workers=num_workers, 
                             pin_memory=True, 
                             shuffle=False)
    
    return train_loader, test_loader

@Timer_Decorator
def attackmodelload(classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False #False：冻结模型的参数，
                                    # 也就是采用该模型已经训练好的原始参数。
                                    #只需要训练我们自己定义的Linear层
    model.fc = nn.Sequential(nn.Linear(num_ftrs,classes),
                                nn.LogSoftmax(dim=1))
    #model.load_state_dict(torch.load('resnet50_acc84.0%.pth'))
    model.load_state_dict(torch.load('resnet50_acc96.0%.pth'))
    model = model.cuda()
    name_ = model._get_name()
    print(f"{name_} loaded")
    

    return model
@Timer_Decorator
def ViT_train_load(classes = 58):
    pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    num_classes = classes
    pretrained_model.head = nn.Linear(pretrained_model.head.in_features, classes)

    model = pretrained_model
    
    model = model.cuda()
    name_ = model._get_name()
    print(f"{name_} loaded")
    return model
    
def ViT_attack_load(classes = 20):
    pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    classes = 58
    pretrained_model.head = nn.Linear(pretrained_model.head.in_features, classes)

    model = pretrained_model
    model.load_state_dict(torch.load('Model\\VisionTransformer\\VisionTransformer_acc99.4%.pth'))
    model = model.cuda()
    name_ = model._get_name()
    print(f"{name_} loaded")
    return model

@Timer_Decorator
def trainmodelload(classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    
    num_ftrs = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False #False：冻结模型的参数，
                                    #也就是采用该模型已经训练好的原始参数。
                                    #只需要训练我们自己定义的Linear层
    model.fc = nn.Sequential(nn.Linear(num_ftrs,classes),
                                nn.LogSoftmax(dim=1))
    #model.load_state_dict(torch.load('resnet50_acc84.0%.pth'))
    #model.load_state_dict(torch.load('resnet50_acc96.0%.pth'))
    model = model.cuda()
    name_ = model._get_name()
    print(f"{name_} loaded")
    

    return model

# Test the model on clean dataset
@Timer_Decorator
def test(model, dataloader):
    model.eval()
    correct, total, loss = 0, 0, 0

    with torch.no_grad():
        for (images, labels) in dataloader:
            # Initialize the Weight Transforms

            weights = ResNet50_Weights.DEFAULT
            #preprocess = weights.transforms()
            #images = preprocess(images)#图像预处理
            
            images = images.cuda()
            labels = labels.cuda()
            model = model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total

# Load the log and generate the training line
@Timer_Decorator
def log_generation(save_path,log):
    # Load the statistics in the log
    epochs, train_rate, test_rate = [], [], []
    

    with open(save_path+'\\'+log, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            epochs.append(int(i[0]))
            train_rate.append(float(i[1]))
            test_rate.append(float(i[2]))

    # Generate the success line
    if(epochs == []):
        print("No log found!")
        return            
    plt.figure(num=0)
    plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    plt.savefig(f"{save_path}/patch_attack_success_rate.png")
    plt.close(0)