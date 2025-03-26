import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import torch

def loss_plot(x,y):
    plt.ion() #开启interactive mode 成功的关键函数
    plt.figure(1,figsize = (17,5))

    plt.plot(x,y,'.') # 第次对画布添加一个点，覆盖式的。
        # plt.scatter(t_now, sin(t_now)) 

    plt.draw()#注意此函数需要调用
    plt.pause(0.001)
    return 

def plot_x(x,y):
    plt.figure(1,figsize = (17,5))
    plt.plot(x,y)
    plt.show()
    return 


