#coding:UTF-8
'''
Created on 2019年6月16日-上午11:27:45
author: Gary-W
'''
import torchvision
model = torchvision.models.alexnet(pretrained=False) #我们不下载预训练权重
print(model)

model = torchvision.models.inception_v3(pretrained=False) #我们不下载预训练权重
print(model)

model = torchvision.models.resnet18(pretrained=False) #我们不下载预训练权重
print(model)

if __name__=="__main__":
    pass

