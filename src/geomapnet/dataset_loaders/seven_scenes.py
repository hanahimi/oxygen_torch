#coding:UTF-8
import os
import os.path as osp
import numpy as np
from torch.utils import data
from geomapnet.common.vis_utils import show_batch, show_stereo_batch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import pickle
from geomapnet.common.pose_utils import process_poses
from geomapnet.dataset_loaders.utils import load_image

import sys
sys.path.insert(0, '../..')
from utils.dataio import get_filelist

class SevenScenes(data.Dataset):
    def __init__(self, scene, 
                        data_path, 
                        train, 
                        img_transformer = None,
                        pose_transform = None, 
                        mode = 0, 
                        seed = 7, 
                        real = False,
                        skip_images = False, 
                        vo_lib = 'orbslam'):
        """
        :param scene: 场景名称 ['chess', 'pumpkin', ...]
        :param data_path: 7scenes 数据集根目录 ('../data/deepslam_data/7Scenes')
        :param train: True (训练集数据), False（测试集数据）
        :param img_transformer: 图像数据转换器
        :param pose_transform: 位姿数据转换器
        :param seed: 随机种子
        :param mode: 0:RGB图像, 1:深度图像, 2: [RGB图像, 深度图像]
        :param real: True（从VO-SLAM/integration 获取位姿）（删除）
        :param skip_images: If True, skip loading images and return None instead
        :param vo_lib: Library to use for VO (currently only 'dso')
        """
        self.mode = mode
        self.img_transformer = img_transformer
        self.pose_transform = pose_transform
        self.skip_images = skip_images
        np.random.seed(seed)    # 如果使用相同的num，则每次生成的随机数都相同
        
        # 加载数据路径
        base_dir = data_path
        data_dir = osp.join(base_dir, scene)
        print("base_dir =",base_dir)
        print("data_dir =",data_dir)
        
        # decide which sequences to use
        if train:
            print("Load Train Set")
            split_file = osp.join(data_dir, 'TrainSplit.txt')
        else:
            print("Load Test Set")
            split_file = osp.join(data_dir, 'TestSplit.txt')
        
        with open(split_file, 'r') as f:
            seqs = [line.strip() for line in f if not line.startswith('#')]
            print(seqs)
        
        self.c_imgs = []                            # rgb images paths
        self.d_imgs = []                            # depth images paths
        self.gt_idx = np.empty((0,), dtype=np.int)  # 创建一个空的ndarray数组[]
        self.poses = np.empty((0, 6))               # translation + log quaternion
        
        # 读取所有序列的 pose数据并收集图像路径
        ps_tbl = {}
        vo_stats = {}
        gt_offset = int(0)      # 用于将多段数据组合时，记录数据段的偏移，初始0
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)
            print("loading:", seq_dir)
            pose_file_lst = get_filelist(seq_dir, ".pose.txt")
            frame_idx_arr = np.array(range(len(pose_file_lst)), dtype=np.int)

            # 4x4 转移矩阵T, 平化其中3x3(R), 3x1(T)部分
            pss = [np.loadtxt(pose_file_lst[i]).flatten()[:12] for i in frame_idx_arr]
            """
            np.array和np.asarray都可以将结构数据转化为ndarray，
            区别是,当数据源是ndarray时，array会copy出一个新的副本，占用新的内存，但asarray不会
            """
            ps_tbl[seq] = np.array(pss)
            print(seq,  ps_tbl[seq].shape)
            # 设置该段数据的初始位姿和
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
            self.gt_idx = np.hstack((self.gt_idx, gt_offset + frame_idx_arr))
            gt_offset += len(pose_file_lst)
            print("gt_idx:{:d}~{:d}, gt_offset={:d}".format(self.gt_idx[0], 
                                                            self.gt_idx[-1],
                                                            gt_offset))
            c_imgs = get_filelist(seq_dir, ".color.png")
            d_imgs = get_filelist(seq_dir, ".depth.png")
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)
        
        # 设置要存储的均值和方程文件
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
            std_t = np.ones(3)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
            print(pose_stats_filename)
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)
        
        # 将位姿数据转换为R3-平移向量与R3-对数四元数
        for seq in seqs:
            pss = process_poses(poses_in=ps_tbl[seq], 
                                mean_t=mean_t, 
                                std_t=std_t,
                                align_R=vo_stats[seq]['R'], # 相对旋转起点 
                                align_t=vo_stats[seq]['t'], # 相对平移起点 
                                align_s=vo_stats[seq]['s']
                                )
            # note: 函数中R转q的方法和SLAM14(3.26)不太一样，得到结果虚部互为相反数（应该一个是左手系，一个是右手系）
            self.poses = np.vstack((self.poses, pss))   # 叠加各seq的pose向量
        print("final pose (T+log(q)):",self.poses.shape)
    
    def __len__(self):
        return self.poses.shape[0]
    
    def __getitem__(self, index):
        pose = self.poses[index]
        if self.pose_transform is not None:
            pose = self.pose_transform(pose)

        img = None

        if self.skip_images:
            return img, pose
        
        else:
            if self.mode == 0:    # rgb
                img = load_image(self.c_imgs[index])
                
            elif self.mode == 1:    # depth
                img = load_image(self.d_imgs[index])
                
            elif self.mode == 2:    # rgb, depth
                img_c = load_image(self.c_imgs[index])
                img_d = load_image(self.d_imgs[index])
                img = [img_c, img_d]
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))
            
        if self.img_transformer is not None:
            if self.mode == 2:
                img = [self.img_transformer(i) for i in img]
            else:
                img = self.img_transformer(img)

        return img, pose

def main():
    """
    visualizes the dataset
    """
    seq_name = 'heads'      # 选择数据集
    mode = 0                # 0: RGB, 1:深度图, 2: [RGB, 深度图]
    num_workers = 2
    
    # 图像预处理设置
    transform = transforms.Compose([
                        transforms.Resize(256),         # 将最小边设置为256，等比例resize
                        transforms.CenterCrop(224),     # 中心裁剪224x224
                        transforms.ToTensor(),          # PIL格式图像转换为tensor
                        # 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。即：Normalized_image=(image-mean)/std
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 原始PIL图像到底是啥？
                                            std=[0.229, 0.224, 0.225])
                                            ])
        
    my_dataset = SevenScenes(seq_name, 
                             r'D:\Ayumi\workspace\data\sevens', 
                             True,    # 返回train数据
                             img_transformer = transform,
                             mode = mode)
    print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq_name, len(my_dataset)))

    if mode < 2:
        img_tensor, pose = my_dataset[10]
        print(img_tensor, pose)
#         print(img_tensor.max(), img_tensor.min()) # 2.5 ~ -2
    
    
    from geomapnet.common.vis_utils import show_batch, show_stereo_batch
    from torchvision.utils import make_grid
    
    data_loader = data.DataLoader(my_dataset, 
                                batch_size=10, 
                                shuffle=True,   # 随机抽取图像
                                num_workers = num_workers)
    batch_cnt = 0
    N = 2
    for batch in data_loader:
        print("mini batch {:d}".format(batch_cnt))
        if mode < 2:
            show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
        elif mode == 2:
            lb = show_batch(make_grid(batch[0][0], nrow=1, padding=25, normalize=True))
            rb = show_batch(make_grid(batch[0][1], nrow=1, padding=25, normalize=True))
            show_stereo_batch(lb, rb)
            
        batch_cnt += 1
        if batch_cnt >= N:
            break

    
    
if __name__ == '__main__':
    main()
