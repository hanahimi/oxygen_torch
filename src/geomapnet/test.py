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

# N x T x C x H x W
x = torch.randn(2,3,4,5,6)
s = x.size()
print(s, s[2:])
x = x.view(-1, *s[2:])
# x = x.view(-1, 4,5,6)
print(x.size())

skip = 10
steps = 4
skips = np.random.randint(1, high=skip+1, size=steps-1)
print(skips)

if __name__=="__main__":
    pass

