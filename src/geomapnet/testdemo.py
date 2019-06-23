#coding:UTF-8
'''
Created on 2019年6月18日-上午10:50:38
author: Gary-W
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init        # 调用初始化参数
import numpy as np

# x = torch.randn(1, 2, 3, 3)
# apx = nn.AdaptiveAvgPool2d((1,1))
# print(x)
# print(apx(x))
# apx2 = nn.AdaptiveAvgPool2d(1)
# print(apx2(x))

# xyz = torch.tensor([[1,1,1],[2,2,2]])
# yuv = torch.tensor([[-1,-1,-1],[-2,-2,-2]])
# wpqr = torch.tensor([[4,4,4,4],[5,5,5,5]])
# cat0 = torch.cat((xyz, yuv), 0)
# cat1 = torch.cat((xyz, wpqr), 1)
# print(xyz)
# print(wpqr)
# print(cat0)
# print(cat1)


"""
# N x T x C x H x W
x = torch.randn(2,3,4,5,6)
s = x.size()
print(s, s[2:])
x = x.view(-1, *s[2:])
# x = x.view(-1, 4,5,6)
print(x.size())
"""
"""
skip = 10
steps = 4
skips = np.random.randint(4, high=10, size=5)
print(skips)
"""

"""
import sys
sys.path.insert(0, '../src/geomapnet')
from geomapnet.common.pose_utils import *
v1 = torch.tensor([[1.0, 2.0, 3.0]])
print(v1)
v2 = torch.tensor([[3.0, 1.0, 3.0]])
print(v2)
v = vdot(v1, v2)
print(v)
"""

# register_hook test
# 由于pytorch会自动舍弃图计算的中间结果，所以想要获取这些数值就需要使用钩子函数。
# 钩子函数包括Variable的钩子和nn.Module钩子，用法相似

grad_list = []

def print_grad(grad):
    grad_list.append(grad)
x = torch.randn(2, 1, requires_grad=True)
print(x)
y = x + 2
print(y)
z = torch.mean(torch.pow(y, 2))
print(z)
lr = 1e-3
y.register_hook(print_grad)
z.backward()
x.data -= lr*x.grad.data
print(grad_list)

# register_forward_hook & register_backward_hook

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
   
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = LeNet()
img = torch.autograd.Variable((torch.arange(32*32*1).view(1,1,32,32)))
img = img.float()

def hook(module, inputdata, output):
    '''把这层的输出拷贝到features中'''
    print(output.data)

handle = net.conv2.register_forward_hook(hook)
print(net(img))

handle.remove()








if __name__=="__main__":
    pass

