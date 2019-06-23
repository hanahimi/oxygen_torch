#-*-coding:UTF-8
"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Composite data-loaders derived from class specific data loaders
"""
import torch
from torch.utils import data
import numpy as np
from geomapnet.common.pose_utils import calc_vos_simple, calc_vos_safe

class MF(data.Dataset):
    """
    Returns multiple consecutive frames, and optionally VOs
    """
    def __init__(self, 
                 dataset, 
                 steps=2,
                 skip=1,
                 variable_skip=False,
                 real=False,
                 train=True,
                 include_vos=False,
                 no_duplicates=False,
                 vo_func=None):
        """
        :param steps: Number of frames to return on every call
        :param skip: Number of frames to skip
        :param variable_skip: If True, skip = [1, ..., skip]
        :param include_vos: True if the VOs have to be appended to poses. If real
        and include_vos are both on, it gives absolute poses from GT and VOs from
        the SLAM / DSO
        :param no_duplicates: if True, does not duplicate frames when len(self) is
        not a multiple of skip*steps
        """
        self.steps = steps
        self.skip = skip
        self.variable_skip = variable_skip
        self.real = real
        self.include_vos = include_vos
        self.train = train
        if vo_func == None:
            self.vo_func = calc_vos_simple
        else:
            self.vo_func = vo_func
            
        self.no_duplicates = no_duplicates

        if dataset == 'sevens':
            from geomapnet.dataset_loaders.seven_scenes import SevenScenes
            self.dset = SevenScenes(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = SevenScenes(*args, skip_images=True, real=False,
                    **kwargs)
        else:
            raise NotImplementedError

        self.L = self.steps * self.skip

    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
        else:
            skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()
        offsets -= offsets[len(offsets) / 2]
        if self.no_duplicates:
            offsets += self.steps/2 * self.skip
        offsets = offsets.astype(np.int)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        """
        :param index: 
        :return: imgs: STEPS x 3 x H x W
                         poses: STEPS x 7
                         vos: (STEPS-1) x 7 (only if include_vos = True)
        """
        idx = self.get_indices(index)
        clip    = [self.dset[i] for i in idx]

        imgs    = torch.stack([c[0] for c in clip], dim=0)
        poses = torch.stack([c[1] for c in clip], dim=0)
        if self.include_vos:
            # vos = calc_vos_simple(poses.unsqueeze(0))[0] if self.train else \
            #     calc_vos_safe(poses.unsqueeze(0))[0]
            vos = self.vo_func(poses.unsqueeze(0))[0]
            if self.real:    # absolute poses need to come from the GT dataset
                clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
                poses = torch.stack([c[1] for c in clip], dim=0)
            poses = torch.cat((poses, vos), dim=0)

        return imgs, poses

    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L



