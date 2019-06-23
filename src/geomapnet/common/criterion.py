#coding:UTF-8
"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
This module implements the various loss functions (a.k.a. criterions) used
in the paper
"""


"""
关于：torch.nn.Parameter()
把这个函数可以理解为类型转换函数，将一个不可训练的类型Tensor
转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
"""

from geomapnet.common import pose_utils
import torch
from torch import nn

class QuaternionLoss(nn.Module):
    """
    Implements distance between quaternions as mentioned in
    D. Huynh. Metrics for 3D rotations: Comparison and analysis
    """
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, q1, q2):
        """
        :param q1: N x 4
        :param q2: N x 4
        :return: 
        """
        loss = 1 - torch.pow(pose_utils.vdot(q1, q2), 2)
        loss = torch.mean(loss)
        return loss

class PoseNetCriterion(nn.Module):
    def __init__(self, 
                 t_loss_fn=nn.L1Loss(), 
                 q_loss_fn=nn.L1Loss(), 
                 sax=0.0,
                 saq=0.0, 
                 learn_beta=False):
        
        super(PoseNetCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 6
        :param targ: N x 6
        :return: 
        """
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + \
            self.sax +\
         torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) +\
            self.saq
        return loss

class MapNetCriterion(nn.Module):
    def __init__(self, 
                 t_loss_fn=nn.L1Loss(), 
                 q_loss_fn=nn.L1Loss(), 
                 sax=0.0,
                 saq=0.0,
                 srx=0, 
                 srq=0.0, 
                 learn_beta=False, 
                 learn_gamma=False):
        """
        Implements L_D from eq. 2 in the paper
        :param t_loss_fn: loss function to be used for translation
        :param q_loss_fn: loss function to be used for rotation
        :param sax: absolute translation loss weight
        :param saq: absolute rotation loss weight
        :param srx: relative translation loss weight
        :param srq: relative rotation loss weight
        :param learn_beta: learn sax and saq?
        :param learn_gamma: learn srx and srq?
        """
        super(MapNetCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        """
        :param pred: N x T x 6
        :param targ: N x T x 6
        :return:
        """

        # absolute pose loss
        s = pred.size()
        abs_loss =\
            torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],
                                                  targ.view(-1, *s[2:])[:, :3]) + \
            self.sax + \
            torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                                  targ.view(-1, *s[2:])[:, 3:]) + \
            self.saq

        # get the VOs
        pred_vos = pose_utils.calc_vos_simple(pred)
        targ_vos = pose_utils.calc_vos_simple(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = \
            torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                                  targ_vos.view(-1, *s[2:])[:, :3]) + \
            self.srx + \
            torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                                  targ_vos.view(-1, *s[2:])[:, 3:]) + \
            self.srq

        # total loss
        loss = abs_loss + vo_loss
        return loss


