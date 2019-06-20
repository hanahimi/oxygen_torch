"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init        # ���ó�ʼ������
import numpy as np


class PoseNet(nn.Module):
    def __init__(self, feature_extractor, 
                        droprate=0.5, 
                        pretrained=True,
                        feat_dim=2048
                        , filter_nans=False):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # �滻feature_extractor��������һ��FC��
        self.new_feature_extractor = feature_extractor
        self.new_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)    # ����ԭ��avgpool�����ã�����Ϊ�µ�����
        ori_fe_out_dim = self.new_feature_extractor.fc.in_features
        # new fc layer
        self.new_feature_extractor.fc = nn.Linear(ori_fe_out_dim, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)   # ����ʹ��log(q)��ʾ�Ƕ�

        # ��ȡ��Ҫ����ѵ���Ĳ�
        if pretrained:
            # ����Ԥѵ�����ݣ�����¼����layer���г�ʼ��
            init_modules = [self.new_feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            # ���򣬳�ʼ����������
            init_modules = self.modules()

        # ������Ҫ��ʼ���Ĳ㣬ʹ��initģ���ʼ��
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)  # fun_ ��ʾinplace����
                if m.bias is not None:  # ������ƫ�����ʼ��Ϊ����0
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.new_feature_extractor(x)   # ʹ��Ǩ�ƹ���������ȡFC����
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)   # ���ո���p��FC�������������Ϊ0
        xyz  = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)    # �ϲ���һ�����
    
    def show_info(self):
        # ��ӡ������������ƺ�״̬
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
        # *s[2:] ���ڰ�s[2:]��torch.Size��������չ������Ϊ����Ĳ���
        eg. 
        x = torch.randn(2,3,4,5,6)
        s = x.size()    # s == torch.Size([2, 3, 4, 5, 6])
        s[2:] Ϊ torch.Size([4, 5, 6])
        *s[2:] �� 4 5 6 �Ƿֱ�3������
        x = x.view(-1, *s[2:]) �ȼ��� x.view(-1, 4,5,6)
        """
        poses = self.mapnet(x)
        poses = poses.view(s[0], s[1], -1)
        return poses
