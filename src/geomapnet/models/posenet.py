"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init        # 调用初始化参数
import numpy as np


class PoseNet(nn.Module):
    def __init__(self, feature_extractor, 
                        droprate=0.5, 
                        pretrained=True,
                        feat_dim=2048
                        , filter_nans=False):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # 替换feature_extractor网络的最后一个FC层
        self.new_feature_extractor = feature_extractor
        self.new_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)    # 不管原来avgpool的设置，都改为新的配置
        ori_fe_out_dim = self.new_feature_extractor.fc.in_features
        # new fc layer
        self.new_feature_extractor.fc = nn.Linear(ori_fe_out_dim, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)   # 文中使用log(q)表示角度

        # 获取需要重新训练的层
        if pretrained:
            # 若有预训练数据，这对新加入的layer进行初始化
            init_modules = [self.new_feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            # 否则，初始化整个网络
            init_modules = self.modules()

        # 遍历需要初始化的层，使用init模块初始化
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)  # fun_ 表示inplace操作
                if m.bias is not None:  # 若存在偏移项，初始化为常数0
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.new_feature_extractor(x)   # 使用迁移过的网络提取FC特征
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)   # 按照概率p将FC特征参数随机设为0
        xyz  = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)    # 合并到一起输出
    
    def show_info(self):
        # 打印网络参数的名称和状态
        params_iter = self.named_parameters()
        for name, param in params_iter:
            print(name," : ",param.shape)
        
class MapNet(nn.Module):
    """
    Implements the MapNet model (green block in Fig. 2 of paper)
    """
    def __init__(self, mapnet):
        """
        :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
        of paper). Not to be confused with MapNet, the model!
        """
        super(MapNet, self).__init__()
        self.mapnet = mapnet

    def forward(self, x):
        """
        :param x: image blob (N x T x C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()
        x = x.view(-1, *s[2:])
        """
        # *s[2:] 等于把s[2:]（torch.Size）的数据展开，变为分离的参数
        eg. 
        x = torch.randn(2,3,4,5,6)
        s = x.size()    # s == torch.Size([2, 3, 4, 5, 6])
        s[2:] 为 torch.Size([4, 5, 6])
        *s[2:] 即 4 5 6 是分别3个参数
        x = x.view(-1, *s[2:]) 等价于 x.view(-1, 4,5,6)
        """
        poses = self.mapnet(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
