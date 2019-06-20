#coding:UTF-8
'''
Created on 2019年6月15日-上午9:17:22
author: Gary-W
'''

"""
1. 读取和归一化 CIFAR10
使用torchvision可以非常容易地加载CIFAR10。
torchvision的输出是[0,1]的PILImage图像，我们把它转换为归一化范围为[-1, 1]的张量。
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# transform模组通过输入转换类的列表构建转换方法
# input[channel] = (input[channel] - mean[channel]) / std[channel]
my_transformer = transforms.Compose(
                        [transforms.ToTensor(),     # pil/np格式图像转换为tensor
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 设置通道的均值和方差，将[0,1]归一化到[-1,1]
                        ])

# 自动下载数据，设为训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = my_transformer)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)  # 设置为单线程加载后（num_workers=0）IDE可以运行
# num_workers=2时只有在jupter里面可以运行
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform = my_transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取随机数据
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图像
imshow(torchvision.utils.make_grid(images))
# 显示图像标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

 
import torch.nn as nn
import torch.nn.functional as F
 
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
net = Net()
 
import torch.optim as optim
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
 
 
for epoch in range(2):  # 多批次循环
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        print(inputs.shape, labels.shape)
        # 梯度置0
        optimizer.zero_grad()
 
        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
 
print('Finished Training')


if __name__=="__main__":
    pass

