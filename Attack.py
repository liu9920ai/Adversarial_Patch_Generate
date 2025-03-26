import torch
import torchvision

from torchvision.models import resnet50

from colorama import Fore, init
from tqdm import trange
import argparse
import csv
import os
import numpy as np
import time

from patch_utils import*
from utils import*

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
parser.add_argument('--train_size', type=int, default=900, help="number of training images")
parser.add_argument('--test_size', type=int, default=100, help="number of test images")

parser.add_argument('--noise_percentage', type=float, default=0.06, help="percentage of the patch size compared with the image size")
parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
parser.add_argument('--lr', type=float, default=0.8, help="learning rate")
parser.add_argument('--max_iteration', type=int, default=100, help="max iteration")
parser.add_argument('--target', type=int, default=0, help="target label")
parser.add_argument('--epochs', type=int, default=20, help="total epoch")
parser.add_argument('--classes', type=int, default=20, help="number of classes")

parser.add_argument('--data_dir', type=str, default='image_TSRD', help="dir of the dataset")#..\Dataset\image
parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")#GPU编号
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
parser.add_argument('--patch_save', type=str, default='patch_save', help='dir of the log')

args = parser.parse_args()


# Patch attack via optimization
# 补丁攻击
# According to reference [1], one image is attacked each time
# 每次攻击一张图片
# Assert: applied patch should be a numpy
# 输入：image: 待攻击图片，applied_patch: 已经贴上的补丁，mask: 补丁的mask，target: 目标标签，probability_threshold: 最小目标概率，model: 模型，lr: 学习率，max_iteration: 最大迭代次数
# Return the final perturbated picture and the applied patch. Their types are both numpy
# 返回最终的扰动图片和贴上的补丁，它们的类型都是numpy
@Timer_Decorator
def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0

    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))

    while target_probability < probability_threshold and count < max_iteration:
        
        count += 1
        # 优化补丁
        # perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        perturbated_image = perturbated_image.requires_grad_(True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # 贴上补丁并测试
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    print(f'补丁共迭代[{count}]次')
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

@Timer_Decorator
def show_image(tensor_image,label,pre,cla,img_num,epoch, path):
    # 将Tensor转换为NumPy数组
    numpy_image = tensor_image.cpu().numpy()
    if numpy_image.shape[0]==1:
        numpy_image = numpy_image[0]

    # 调整数组维度和坐标轴顺序
    numpy_image = np.transpose(numpy_image, (1, 2, 0))
    
    # 归一化图像数据
    min_val = np.min(numpy_image)
    max_val = np.max(numpy_image)
    normalized_image = (numpy_image - min_val) / (max_val - min_val)
    path = f"{path}/ori_and_patch"
    if not os.path.exists(path):
            os.makedirs(path)
    
    # 显示图像
    plt.imshow(normalized_image)
    plt.axis('off')
    name = f"num{img_num}_{cla}_label{label}_pre{pre}_epoch{epoch}.png"
    plt.savefig(f"{path}/num{img_num}_{cla}_label{label}_pre{pre}_epoch{epoch}.png",
                bbox_inches='tight',
                pad_inches=0.0)
    
    print(f'image {name} saved')
    plt.close()

def main(noise):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    image_size = (3, 224, 224)
    args.noise_percentage = noise
    # Load the model
    
    #model = attackmodelload(args.classes)
    model = ViT_attack_load()
    model.train()

    # Load the datasets
    train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers)
    
    # Test the accuracy of model on trainset and testset
    trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
    print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

    # Initialize the patch
    patch = patch_initialization(args.patch_type, image_size, noise_percentage=args.noise_percentage)
    print('The shape of the patch is', patch.shape)

    with open(args.log_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_success", "test_success"])

    best_patch_epoch, best_patch_success_rate = -1, 0
    worst_patch_epoch, worst_patch_success_rate = -1, 1
    
    patch_save = f'{args.patch_save}\\noise{args.noise_percentage}'
    
    image_num = 0
    # Generate the patch
    for epoch in range(args.epochs):
        l_t = time.time()#本轮开始时间
        acc_continue_num = 0
        print(f"epoch{epoch+1}/{args.epochs}",end = '')
        #本轮用时
        
        train_total, train_actual_total, train_success = 0, 0, 0
        i = 1
        for (image, label) in train_loader:
            #start_ = time.time()
            print(Fore.RESET + f'epoch{epoch+1}/{args.epochs} image{i}/{args.train_size} label:{int(label)} ')
            i+=1
            train_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()# image is the input of the model
            label = label.cuda()# label is the true label of the image
            output = model(image)# output is the prediction of the model
            predicted = torch.max(output.data, 1).indices
            predicted = int(predicted)
            
            
            
            k = int(predicted) == int(label)
            
            
            flag = '成功' if k else '失败'
            color = Fore.GREEN if k else Fore.RED
            print(f'预测结果:{int(predicted)}, 目标:{args.target},预测是否成功:',color + f'{flag}' + Fore.RESET)
            
            
            train_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size)
            perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
            perturbated_image = torch.from_numpy(perturbated_image).cuda()
            
            output = model(perturbated_image)
            predicted0 = torch.max(output.data,1).indices
            predicted0 = int(predicted0)
            # 每100张图片保存一次图片及其对应打补丁图片
            if image_num % 100 == 0:
                show_image(image,int(label),predicted,'ori',image_num,epoch,patch_save)
                show_image(perturbated_image,int(label),predicted0,'patch',image_num,epoch,patch_save)


            print(Fore.RESET + f'攻击结果:{int(predicted0)}, ',end = '')
            if int(predicted0) == args.target:
                train_success += 1
                print(Fore.GREEN + '攻击成功^_^'+Fore.RESET)
            else:
                print(Fore.RED + '攻击失败\+_+/'+Fore.RESET)
            #更新patch
            patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
            

            image_num+=1

        
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        # patch = np.transpose(patch, (1, 2, 0))
        # plt.imshow(patch)
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
        # plt.imshow(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1)
        
        if not os.path.exists(patch_save):
            os.makedirs(patch_save)
        plt.savefig(patch_save + "/" + str(epoch+1) + " patch.png")
        plt.close()
        
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch+1, 100 * train_success / train_actual_total))
        train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch+1, 100 * train_success_rate))
        test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
        print("Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch+1, 100 * test_success_rate))

        # Record the statistics
        with open(patch_save + "\\" + args.log_dir, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_success_rate, test_success_rate])
            
        if test_success_rate < worst_patch_success_rate:
            worst_patch_success_rate = test_success_rate
            worst_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            plt.savefig(f"{patch_save}/worst_patch.png")

        if test_success_rate > best_patch_success_rate:
            best_patch_success_rate = test_success_rate
            best_patch_epoch = epoch
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            plt.savefig(f"{patch_save}/best_patch.png")

        # Load the statistics and generate the line
        log_generation(patch_save,args.log_dir)
        print(f'epoch{epoch+1} finished')
        print(f' time of this epoch:{time.time()-l_t}s')
        
        
    print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))
 

if __name__ == '__main__':
    
    noises = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    
    noises = [0.01]
    
    noises.reverse()
    #print (noises)
    for noise in noises:
        main(noise)
        

