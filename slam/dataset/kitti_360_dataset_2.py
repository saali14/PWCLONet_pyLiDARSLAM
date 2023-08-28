

import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader
import torch
import time

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

from tqdm import tqdm

import open3d

import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)


from slam.common.kitti360_utils import KITTI360_IO, KITTI360_TRANSFORMATIONS, KITTI360_TOOLS, KITTI360_INFOS
from slam.dataset.configuration import DatasetConfig



###########################################################################
### DON'T FORGET TO CHANGE THE `DATASET` CLASS IN `dataset.__init__.py` ###
###########################################################################


@dataclass
class KITTI360Config(DatasetConfig):
    """A configuration object read from a yaml conf"""
    # -------------------
    # Required Parameters
    root_dir: str = MISSING
    dataset: str = "kitti_360"

    # ------------------------------
    # Parameters with default values
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 3
    down_fov: int = -24

    train_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7])
    eval_sequences: list = field(default_factory=lambda: [9, 10])
    test_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 9, 10])

    num_points: int = -1
    scalar_last: bool = MISSING
    velo_to_pose: bool = MISSING

    augment: bool = True
    frame_gap: int = MISSING
    train_frame_gap: int = MISSING
    
    near_treshold: float = MISSING


class Kitti360Dataset(Dataset):
    def __init__(self, config: KITTI360Config, is_training: bool = True, is_testing: bool = False, random_seed = 3):
        """
        Arguments:
            root_path (string): Path to the dataset directory that should contain 'data_3d_raw' and 'data_poses'.
            npoints (int): Numbre of points to keep for each point cloud.
            is_training (bool): whether to load training dataset or evaluation dataset.
        """

        self.config = config
        self.npoints = config.num_points
        self.datapath = config.root_dir

        self.random_seed = random_seed
        self.is_training = is_training
        self.is_testing = is_testing
                
        self.sequence_size = {
            0: 11518,
            2: 14849, # 19240,
            3: 1031,
            4: 11587,
            5: 6743,
            6: 9699,
            7: 3396,
            9: 14056,
            10: 3836
        }

        if self.is_training:
            sequences = self.config.train_sequences
        elif self.is_testing: 
            sequences = self.config.test_sequences
        else: 
            sequences = self.config.eval_sequences

        self.len_list = [0]
        for i in range(len(sequences)):
            self.len_list.append(self.len_list[i] + self.sequence_size[sequences[i]])
        self.file_map = sequences

        self.poses = {}
        self.save_dir = 'GT'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def filter_pcd(self, points):
        wheel_axis_z = -(KITTI360_INFOS.VELODYNE_HEIGHT - KITTI360_INFOS.WHEEL_AXIS_HEIGHT)
        is_ground = points[:,2] < wheel_axis_z
        not_ground = np.logical_not(is_ground)

        near_mask_x = np.logical_and(points[:, 0] < self.config.near_treshold, points[:, 0] > -self.config.near_treshold)
        near_mask_y = np.logical_and(points[:, 1] < self.config.near_treshold, points[:, 1] > -self.config.near_treshold) # /!\ is y-axis in index 2 or 1 ???????? 
        near_mask = np.logical_and(near_mask_x, near_mask_y)

        near_mask = np.logical_and(not_ground, near_mask)
        indices = np.where(near_mask)[0]

        if len(indices) >= self.npoints:
            sample_idx = np.random.choice(indices, self.npoints, replace=False)
        elif len(indices) > 0:
            sample_idx = np.concatenate((indices, np.random.choice(indices, self.npoints - len(indices), replace=True)), axis=-1)
        else:
            warnings.warn(f'Filterting results on empty point cloud. Proceed to random selection')
            sample_idx = np.random.choice(np.arange(len(points)), self.npoints, replace=True)

        points = points[sample_idx, :]

        return points


    def __len__(self):
        return self.len_list[-1]


    def __getitem__(self, index):

        #if not self.is_training:
        #    np.random.seed(self.random_seed)

        if self.is_training:
            frame_gap = np.random.randint(1, self.config.train_frame_gap+1)
        else:
            frame_gap = np.random.randint(1, self.config.frame_gap+1)

        for seq_idx, seq_num in enumerate(self.len_list):
            if index < seq_num:
                cur_seq_idx = seq_idx - 1
                cur_seq = self.file_map[cur_seq_idx]
                cur_idx_pc2 = index - self.len_list[seq_idx-1]

                if cur_idx_pc2 < frame_gap:
                    cur_idx_pc1 = 0
                else:
                    cur_idx_pc1 = cur_idx_pc2 - frame_gap        ####### 1 frame gap  #######   
                break     

        drive_name = KITTI360_IO.drive_foldername(cur_seq)

        # ----- load point clouds -----

        cur_lidar_dir = os.path.join(self.datapath, 'data_3d_raw', drive_name)
        pc1_bin = os.path.join(cur_lidar_dir, 'velodyne_points', 'data')
        pc2_bin = os.path.join(cur_lidar_dir, 'velodyne_points', 'data')
        point1 = KITTI360_IO.load_raw_3D_pcd(cur_idx_pc1, pc1_bin)
        point2 = KITTI360_IO.load_raw_3D_pcd(cur_idx_pc2, pc2_bin)

        if point1.shape[0] < point2.shape[0]:
            n = point1.shape[0]
        else:
            n = point2.shape[0] # num points of pc1 and pc2

        point1 = point1[:n, :3] # :3
        point2 = point2[:n, :3] # :3

        point1 = self.filter_pcd(point1)
        point2 = self.filter_pcd(point2)

        n = point1.shape[0]
        
        # ----- load relative pose -----
        pose = KITTI360_IO.get_sequence_poses(self.datapath, cur_seq, velo_to_world=False, relative=True)

        T_diff = np.eye(4)
        for i in range(frame_gap):
            frame = cur_idx_pc1 + i + 1
            if frame > cur_idx_pc2:
                break
            curr_pose = pose[frame, :].reshape((4,4))
            T_diff = T_diff @ curr_pose 
        

        if self.config.augment and (self.is_training == True):  ###  augment
            # transform the point clouds using Tr_diff plus an additional augmentation
            
            anglex = np.clip(0.01* np.random.randn(),-0.02, 0.02).astype(np.float32)* np.pi / 4.0
            angley = np.clip(0.05* np.random.randn(),-0.1, 0.1).astype(np.float32)* np.pi / 4.0
            anglez = np.clip(0.01* np.random.randn(),-0.02, 0.02).astype(np.float32)* np.pi / 4.0

            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)
            Rx = np.array([[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]])
            
            R_trans = Rx.dot(Ry).dot(Rz)

            xx = np.clip(0.1* np.random.randn(),-0.2, 0.2).astype(np.float32)
            yy = np.clip(0.05* np.random.randn(),-0.15, 0.15).astype(np.float32)
            zz = np.clip(0.5* np.random.randn(),-1, 1).astype(np.float32)

            add_3 = np.array([[xx], [yy], [zz]])

            T_trans = np.concatenate([R_trans, add_3], axis = -1)
            filler = np.array([0.0, 0.0, 0.0, 1.0])
            filler = np.expand_dims(filler, axis = 0)   ##1*4
            T_trans = np.concatenate([T_trans, filler], axis=0)#  4*4
            
            add_T3 = np.ones((n, 1))
            pos2_trans = np.concatenate([point2, add_T3], axis = -1)
            pos2_trans = np.matmul(T_trans, pos2_trans.T)
            point2 = pos2_trans.T[ :, :3]

            T_gt = np.matmul(T_trans, T_diff)
            
            """
            EXPLANATION:
            -------------
            
            --> In kitti360 p2_ is the target and p1 is the source 
            p2 = T_diff * p1 (given)
            p2_ = T_aug * p2 = T_aug * T_diff * p1
            Then T_gt = T_aug * T_diff
            
            --> In kitti p1 is the target and p2_ is the source 
            p1 = T_diff * p2 (given)
            p2_ = T_aug * p2 = T_aug * inv(T_diff) * p1
            p1 = T_diff * inv(T_aug) * p2_
            """

        else:
            # transform the point cloud using only Tr_diff
            T_gt = T_diff
        
        R_gt = T_gt[:3, :3]
        t_gt = np.squeeze(T_gt[:3, 3])

        q_gt = KITTI360_TOOLS.mat2quat(R_gt, scalar_last=self.config.scalar_last)

        point1_float = point1.astype(np.float32)
        point2_float = point2.astype(np.float32)
        q_gt_float = q_gt.astype(np.float32)
        t_gt_float = t_gt.astype(np.float32)

        # I switched PCD1 and PCD2
        return point1_float, point2_float, q_gt_float, t_gt_float, cur_seq, cur_idx_pc2


def plot_2D(ax, data: dict, fontsize=10, axes=['x', 'y'], plot_radius = None):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        color_bar = False
        for key, value in data.items():
            marker = '.'
            if 'marker' in value.keys():
                marker = value['marker']
            alpha = 1
            if 'alpha' in value.keys():
                alpha = value['alpha']
            cmap = None
            if 'cmap' in value.keys():
                cmap = mpl.colormaps[value['cmap']]
                color_bar = True
            if key == 'gt':
                plt.scatter(value['x'], value['y'], c=value['color'], label=key, marker=marker, cmap=cmap, alpha=alpha, linewidths=0.5, zorder=0)
            else:
                plt.plot(value['x'], value['y'], color=value['color'], ls="--", label=key, linewidth=1.5, zorder=10) #linestyle='dashed')

        plt.legend(loc="upper right", prop={'size':fontsize})
        plt.xlabel(f'{axes[0]} (m)', fontsize=fontsize)
        plt.ylabel(f'{axes[1]} (m)', fontsize=fontsize)
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        if plot_radius is None:
            plot_radius = max([abs(lim - mean_)
                                for lims, mean_ in ((xlim, xmean),
                                                    (ylim, ymean))
                                for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        if color_bar:
            plt.colorbar(label='Distance Travelled (m)')

        return plot_radius


def plotPath_2D_3(seq, poses_gt, poses_result, plot_path):
        '''
            plot path in XY, XZ and YZ plane
        '''
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fontsize_ = 10        
        
        ### get the value
        if poses_gt:
            dist = KITTI360_TOOLS.trajectoryDistances(poses_gt)

            if isinstance(poses_gt, dict):
                poses_gt = [(k,poses_gt[k]) for k in sorted(poses_gt.keys())]
            else:
                poses_gt = [(k,poses_gt[k,:,:]) for k in range(poses_gt.shape[0])]
            x_gt = np.asarray([pose[0,3] for _,pose in poses_gt])
            y_gt = np.asarray([pose[1,3] for _,pose in poses_gt])
            z_gt = np.asarray([pose[2,3] for _,pose in poses_gt])

        if isinstance(poses_result, dict):
            poses_result = [(k, poses_result[k]) for k in sorted(poses_result.keys())]
        else:
            poses_result = [(k,poses_result[k,:,:]) for k in range(poses_result.shape[0])]
        x_pred = np.asarray([pose[0,3] for _,pose in poses_result])
        y_pred = np.asarray([pose[1,3] for _,pose in poses_result])
        z_pred = np.asarray([pose[2,3] for _,pose in poses_result])

        
        fig = plt.figure(figsize=(20,6), dpi=100)
        ### plot the figure
        plt.subplot(1,3,1)
        ax = plt.gca()
        data = {
            'pred': {
                'x': x_pred,
                'y': z_pred,
                'color': '#FD3412'
            }
        }
        if poses_gt:
            data['gt'] = {
                'x': x_gt,
                'y': z_gt,
                'color': dist, #gt_color
                'alpha': 1,
                'cmap': 'viridis'
            }
        plot_radius = plot_2D(ax, data, fontsize=fontsize_, axes=['x', 'z'])

        plt.subplot(1,3,2)
        ax = plt.gca()
        data['pred']['x'] = x_pred
        data['pred']['y'] = y_pred
        if poses_gt:
            data['gt']['x'] = x_gt
            data['gt']['y'] = y_gt
        plot_2D(ax, data, fontsize=fontsize_, plot_radius=plot_radius, axes=['x', 'y'])

        plt.subplot(1,3,3)
        ax = plt.gca()
        data['pred']['x'] = y_pred
        data['pred']['y'] = z_pred
        if poses_gt:
            data['gt']['x'] = y_gt
            data['gt']['y'] = z_gt
        plot_2D(ax, data, fontsize=fontsize_, plot_radius=plot_radius, axes=['y', 'z'])

        png_title = "{}_path".format(seq)
        fig.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


def plotPath_3D(seq, poses_gt, poses_result, plot_path):
    """
        plot the path in 3D space
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    start_point = [[0], [0], [0]]
    fontsize_ = 8
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'        

    poses_dict = {}      
    poses_dict["Ours"] = poses_result
    if poses_gt is not None:
        dist = KITTI360_TOOLS.trajectoryDistances(poses_gt)
        poses_dict["Ground Truth"] = poses_gt

    fig = plt.figure(figsize=(8,8), dpi=110)
    ax = plt.axes(projection='3d')
    #ax = fig.gca(projection='3d')

    for key,_ in poses_dict.items():
        if isinstance(poses_dict[key], dict):
            plane_point = []
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                plane_point.append([pose[0,3], pose[1,3], pose[2,3]])
            plane_point = np.asarray(plane_point)
        else:
            plane_point = poses_dict[key][:,:3,3]
        
        style = style_pred if key == 'Ours' else style_gt
        if key == 'Ground Truth':
            p = ax.scatter(plane_point[:,0], plane_point[:,1], plane_point[:,2], c=dist, cmap='viridis', alpha=1, label=key, linewidths=0.5, zorder=0, marker='.')
        else:
            plt.plot(plane_point[:,0], plane_point[:,1], plane_point[:,2], color='r', ls="--", label=key, linewidth=1.5, zorder=10)
            
    if poses_gt is not None:
        fig.colorbar(p, label='Distance Travelled (m)')

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)
    plot_radius = max([abs(lim - mean_)
                    for lims, mean_ in ((xlim, xmean),
                                        (ylim, ymean),
                                        (zlim, zmean))
                    for lim in lims])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    ax.legend()
    ax.set_xlabel('x (m)', fontsize=fontsize_)
    ax.set_ylabel('y (m)', fontsize=fontsize_)
    ax.set_zlabel('z (m)', fontsize=fontsize_)
    ax.view_init(elev=20., azim=-35)

    png_title = "{}_path_3D".format(seq)
    fig.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def progress_bar(dataloader: DataLoader, desc: str = ""):
    return tqdm(enumerate(dataloader, 0),
                desc=desc,
                total=len(dataloader),
                ncols=120, ascii=True, position=0, leave=True)

    
if __name__ == '__main__':
    if 'KITTI360_DATASET' in os.environ:
        root_path = os.environ['KITTI360_DATASET']
    else:
        root_path = '/data/3d_cluster/ADAS_data/KITTI-360/heavy_data'
    npoints = 8192
    is_training = False
    is_testing = True

    config = KITTI360Config()
    config.root_dir = root_path
    config.num_points = 8192
    config.train_sequences = [3]
    config.test_sequences = [0, 3, 7, 9]
    config.scalar_last = False
    config.velo_to_pose = True
    kitti360_dataset = Kitti360Dataset(config = config, is_training = is_training, is_testing = is_testing)

    dataloader = DataLoader(kitti360_dataset, batch_size=1, shuffle=True, num_workers=2)

    print(len(kitti360_dataset))
    print(len(dataloader))

    start = time.time()
    data_progress_bar = progress_bar(dataloader, desc=f"Training epoch n° 1")
    time_log = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
    rel_gt_poses = {}

    for i_batch, sample_batched in data_progress_bar:
        for i in range(sample_batched[4].size(0)):
            curr_seq = sample_batched[4][i].cpu().detach().item()
            frame_idx = sample_batched[5][i].cpu().detach().item()
            
            if curr_seq not in rel_gt_poses.keys():
                rel_gt_poses[curr_seq] = {}

            gt_q = sample_batched[2][i,:].cpu().detach().numpy().reshape(-1)
            gt_t = sample_batched[3][i,:].cpu().detach().numpy().reshape(3)
            R = KITTI360_TOOLS.quat2mat_batch(gt_q, scalar_last=kitti360_dataset.config.scalar_last)
            R = R.reshape(3,3)
            T = gt_t.reshape(3)
            trans = np.eye(4)
            trans[:3,:3] = R
            trans[:3,3] = T
            rel_gt_poses[curr_seq][frame_idx] = trans
    
    abs_gt_poses = {}
    for seq in rel_gt_poses.keys():
        pred_rel_seq_save_path = os.path.join(kitti360_dataset.save_dir, str(seq).zfill(2) + '_pred_rel.txt')
        KITTI360_IO.save_poses(rel_gt_poses[seq], pred_rel_seq_save_path)

        abs_gt_poses[seq] = KITTI360_TRANSFORMATIONS.convert_to_absolute(rel_gt_poses[seq], velo_to_pose=False)#kitti360_dataset.config.velo_to_pose)

        #abs_gt_poses[seq] = KITTI360_TRANSFORMATIONS.velo_to_pose(abs_gt_poses[seq])
        pred_seq_save_path = os.path.join(kitti360_dataset.save_dir, str(seq).zfill(2) + '_pred.txt')
        KITTI360_IO.save_poses(abs_gt_poses[seq], pred_seq_save_path)

        pred_seq_save_dir = os.path.join(kitti360_dataset.save_dir, str(seq).zfill(2))
        if not os.path.exists(pred_seq_save_dir):
            os.makedirs(pred_seq_save_dir)
        
        pred_seq_save_path = os.path.join(pred_seq_save_dir, '3d_path_pred.png')
        plotPath_3D(seq, abs_gt_poses[seq], abs_gt_poses[seq], pred_seq_save_path)

        pred_seq_save_path = os.path.join(pred_seq_save_dir, '2d_path_pred.png')
        plotPath_2D_3(seq, abs_gt_poses[seq], abs_gt_poses[seq], pred_seq_save_path)

    sys.exit(0)

    """import open3d as o3d

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(batch[0])
    pcd2.points = o3d.utility.Vector3dVector(batch[1])
    
    #pcd1.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (len(batch[0]), 1)))
    pcd2.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (len(batch[1]), 1)))

    o3d.visualization.draw_geometries([pcd1])"""
