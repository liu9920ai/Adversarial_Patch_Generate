from utils import *
import torch.nn as nn
from plot import loss_plot
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __name__ == '__main__':

    train_size = 5000
    test_size = 1000
    batch_size = 50
    data_dir = 'image_TSRD'
    num_works = 10
    epoch = 20
    
    train_loader, test_loader = dataloader(train_size,test_size,data_dir,batch_size,6)
    
    model = ViT_train_load(classes = 58)
   
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


    #开始训练模型
    
    img_all = 0
    loss_list = []
    img_num_list = []
    for epoch in range(epoch):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs, labels = data#数据
            inputs, labels = inputs.cuda(), labels.cuda()#数据转移到GPU上
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #loss_plot(img_all,loss.item())
            loss_list.append(loss.item())
            img_num_list.append(img_all)
            img_all+=1
            print(f'epoch:{epoch+1},step:{i+1},img_all{img_all}/{train_size*epoch}({img_all/train_size:.2f}%),loss:{loss.item():.4f}')
    
    

    #训练完毕，测试模型
    model.eval()
    total = 0#总共测试的图片数量
    correct = 0#正确的图片数量
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    #模型准确率
    accuracy = correct/total
    print(f'Accuracy of the network on the 100 test images:{100*correct/total}%')
    
    #如果准确率大于73%，保存模型
    if(accuracy>=0.73):
        name = model._get_name()
        if not os.path.exists(f'Model\\{name}'):
            os.makedirs(f'Model\\{name}')
        
        torch.save(model.state_dict(),f'Model\\{name}\\{name}_acc{accuracy*100}%.pth')
        #画图并保存
        plt.plot(img_num_list,loss_list)
        plt.savefig(f'Model\\{name}\\loss_plot{100*accuracy}%.png')

