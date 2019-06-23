#coding:UTF-8
'''
Created on 2019年6月23日-下午8:01:20
author: Gary-W
'''
import numpy as np
import os.path as osp
import torch
from torch import nn
from torchvision import transforms, models
from geomapnet.models.posenet import PoseNet, MapNet
from geomapnet.dataset_loaders.seven_scenes import SevenScenes
from geomapnet.dataset_loaders.composite import MF
import configparser
import json
from geomapnet.common.criterion import PoseNetCriterion, MapNetCriterion
from geomapnet.common.optimizer import Optimizer

DATASET = "sevens"
SCENE = "heads"
MODLE = "posenet"   # posenet, mapnet
CONFIG_FILE = "configs/{:s}.ini".format(MODLE) 
DEVICE = 0
LEARN_BETA = True
LEARN_GAMMA = True

"""
load cfg
"""
settings = configparser.ConfigParser()
with open(CONFIG_FILE, 'r') as f:
    settings.read_file(f)
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)  #  颜色数据增强

sax = 0.0
saq = section.getfloat('beta')
if MODLE.find('mapnet') >= 0:
    skip = section.getint('skip')
    real = section.getboolean('real')
    variable_skip = section.getboolean('variable_skip')
    srx = 0.0
    srq = section.getfloat('gamma')
    steps = section.getint('steps')

section = settings['optimization']
opt_method = section['opt']
print("opt_method = {:s}".format(opt_method))
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
print("opt_method =", optim_config)
lr = optim_config.pop('lr') # json 自动转换为float
weight_decay = optim_config.pop('weight_decay')

"""
base model
"""
feature_extractor = models.resnet18(pretrained=True)
posenet = PoseNet(feature_extractor, 
                  droprate = dropout, 
                  pretrained = True,
                  filter_nans=(MODLE=='mapnet++'))

"""
build model and set loss function
"""
if MODLE == 'posenet':
    model = posenet
    train_criterion = PoseNetCriterion(sax=sax, 
                                       saq=saq, 
                                       learn_beta=LEARN_BETA)
    val_criterion = PoseNetCriterion()
elif MODLE.find('mapnet') >= 0:
    model = MapNet(mapnet=posenet)
    train_criterion = MapNetCriterion(sax=sax, 
                                      saq=saq, 
                                      srx=srx, 
                                      srq=srq, 
                                      learn_beta=LEARN_BETA,
                                      learn_gamma=LEARN_GAMMA)
    val_criterion = MapNetCriterion()
    
else:
    raise NotImplementedError
print("build model:{:s}".format(MODLE))

"""
build optimizer, 加入同方差变量到训练参数中
"""
# note: 此法添加sax，saq不会在model.parameters()或model.named_parameters()
# 而是在train_criterion的parameters()里面
param_list = [{'params': model.parameters()}]
if LEARN_BETA \
    and hasattr(train_criterion, 'sax') \
    and hasattr(train_criterion, 'saq'):
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})

if LEARN_GAMMA \
    and hasattr(train_criterion, 'srx') \
    and hasattr(train_criterion, 'srq'):
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
for name, param in train_criterion.named_parameters():
    print(name, param.shape)
optimizer = Optimizer(params=param_list, 
                      method=opt_method, 
                      base_lr=lr,
                      weight_decay=weight_decay, **optim_config)

"""
transformers
"""
data_dir = r"D:/Ayumi/workspace/data/{:s}".format(DATASET)
stats_file = osp.join(data_dir, SCENE, 'stats.txt')
stats = np.loadtxt(stats_file)
print("mean & std:\n",stats)
crop_size = (224, 224)
tforms = [transforms.Resize(256)]
if color_jitter > 0:
    assert color_jitter <= 1.0
    print('Using ColorJitter data augmentation', color_jitter)
    tforms.append(transforms.ColorJitter(brightness=color_jitter,
                                         contrast=color_jitter, 
                                         saturation=color_jitter, 
                                         hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], 
                                   std=np.sqrt(stats[1])))
# 将多个图像转换器组合到一起
img_transform = transforms.Compose(tforms)
# 将位姿转换为torch张量
pose_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

"""
datasets
"""
if MODLE == 'posenet':
    if DATASET == 'sevens':
        kwargs = dict(scene=SCENE, 
                      data_path=data_dir,
                      img_transformer=img_transform,
                      pose_transform=pose_transform,
                      seed=7)
        train_set = SevenScenes(train=True, **kwargs)
        val_set = SevenScenes(train=False, **kwargs)
    else:
        raise NotImplementedError
    
# elif MODLE.find('mapnet') >= 0:
#     kwargs = dict(scene=SCENE, 
#                   data_path=data_dir,
#                   img_transformer=img_transform,
#                   pose_transform=pose_transform,
#                   seed=7,
#                   dataset=SCENE, 
#                   skip=skip, 
#                   steps=steps,
#                   variable_skip=variable_skip)
#     train_set = MF(train=True, real=real, **kwargs)
#     val_set = MF(train=False, real=real, **kwargs)


if __name__=="__main__":
    pass

