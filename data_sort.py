import os
import shutil
import csv
import tqdm

def openread(file_name):
    with open(file_name, 'r') as f:
        data = csv.reader(f)
        data = list(data)
        data = data[1:]
        for id,label in data:
            id = int(id)
            label = int(label)
            
        
    return data

data = openread('cifar10_tiny/trainLabels.csv')

#给数据集分类

old_path = 'cifar10_tiny/train'
new_path = 'image'
#分类好的图片文件夹所存放的位置，每一类图片存放到同一个文件夹中

for id,label in data:
    if not os.path.exists(os.path.join(new_path,label)):
        os.makedirs(os.path.join(new_path,label))
    shutil.copy(os.path.join(old_path,str(id)+'.png'),os.path.join(new_path,label,str(id)+'.png'))