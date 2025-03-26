from utils import *
import torch.nn as nn
from plot import loss_plot
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_size = 900
    test_size = 100
    batch_size = 50
    data_dir = 'image'
    num_works = 10
    epoch = 20

    train_loader, test_loader = dataloader(train_size,test_size,data_dir,batch_size,6)

    resnet50 = trainmodelload(classes = 20)

    resnet50 = resnet50.cuda()
    
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.fc.parameters(),lr=0.0001)


    #开始训练模型
    img_all = 0
    for epoch in range(epoch):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs, labels = data#数据
            inputs, labels = inputs.cuda(), labels.cuda()#数据转移到GPU上
            outputs = resnet50(inputs)
            loss = criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_plot(img_all,loss.item())
            img_all+=1
            print(f'epoch:{epoch+1},step:{i+1},img_all{img_all},loss:{loss.item()}')
    

    #训练完毕，测试模型
    resnet50.eval()
    total = 0#总共测试的图片数量
    correct = 0#正确的图片数量
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = resnet50(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    #模型准确率
    accuracy = correct/total
    print(f'Accuracy of the network on the 100 test images:{100*correct/total}%')
    #如果准确率大于73%，保存模型
    if(accuracy>=0.73):
        torch.save(resnet50.state_dict(),f'Model\\ResNet50\\resnet50_acc{accuracy*100}%.pth')

