

import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader
import torch
import time

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

from tqdm import tqdm
import math

import open3d
from pathlib import Path

from scipy.spatial.transform.rotation import Rotation as R, Slerp
from scipy.interpolate import interp1d

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

def load_raw_3D_pcd(frame: int, raw3DPcdPath: str) -> np.ndarray:
    pcdFile = os.path.join(raw3DPcdPath, '%06d.bin' % frame)
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4])
    return pcd


def read_calib_file(path):  # changed
    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


@dataclass
class KittiOdometryConfig(DatasetConfig):
    """A configuration object read from a yaml conf"""
    # -------------------
    # Required Parameters
    root_dir: str = MISSING
    dataset: str = "kitti"

    # ------------------------------
    # Parameters with default values
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 3
    down_fov: int = -24

    train_sequences: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    eval_sequences: list = field(default_factory=lambda: [7, 8, 9, 10])
    test_sequences: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    num_points: int = -1
    scalar_last: bool = MISSING
    velo_to_pose: bool = MISSING

    augment: bool = MISSING
    frame_gap: int = MISSING
    train_frame_gap: int = MISSING


class KittiOdometryDataset(Dataset):
    def __init__(self, config: KittiOdometryConfig, is_training: bool = True, is_testing: bool = False, random_seed = 3):
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
            0: 4541,
            1: 1101,
            2: 4661,
            3: 801,
            4: 271,
            5: 2761,
            6: 1101,
            7: 1101,
            8: 4071,
            9: 1591,
            10: 1201
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

        if self.config.frame_gap < 1:
            self.config.frame_gap = 1


    def filter_pcd(self, points):
        # points have axis z=front, x=left, y=up
        is_ground = points[:,1] > 1.1

        not_ground = np.logical_not(is_ground)
        
        near_mask_x = np.logical_and(points[:, 0] < 30, points[:, 0] > -30)
        near_mask_y = np.logical_and(points[:, 2] < 30, points[:, 2] > -30)            
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
    

    def euler2mat(self, anglex, angley, anglez):

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

        return R_trans


    def mat2euler(self, M, cy_thresh=None, seq='zyx'):

        M = np.asarray(M)
        if cy_thresh is None:
            cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33*r33 + r23*r23)
        if seq=='zyx':
            if cy > cy_thresh: # cos(y) not close to zero, standard form
                z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else: # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21,  r22)
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = 0.0
        elif seq=='xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi/2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi/2
        else:
            raise Exception('Sequence not recognized')
        return z, y, x


    def euler2quat(self, z=0, y=0, x=0, isRadian=True):
        ''' Return quaternion corresponding to these Euler angles
        Uses the z, then y, then x convention above
        Parameters
        ----------
        z : scalar
            Rotation angle in radians around z-axis (performed first)
        y : scalar
            Rotation angle in radians around y-axis
        x : scalar
            Rotation angle in radians around x-axis (performed last)
        Returns
        -------
        quat : array shape (4,)
            Quaternion in w, x, y z (real, then vector) format
        Notes
        -----
        We can derive this formula in Sympy using:
        1. Formula giving quaternion corresponding to rotation of theta radians
            about arbitrary axis:
            http://mathworld.wolfram.com/EulerParameters.html
        2. Generated formulae from 1.) for quaternions corresponding to
            theta radians rotations about ``x, y, z`` axes
        3. Apply quaternion multiplication formula -
            http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
            formulae from 2.) to give formula for combined rotations.
        '''
    
        if not isRadian:
            z = ((np.pi)/180.) * z
            y = ((np.pi)/180.) * y
            x = ((np.pi)/180.) * x
        z = z/2.0
        y = y/2.0
        x = x/2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        return np.array([
                        cx*cy*cz - sx*sy*sz,
                        cx*sy*sz + cy*cz*sx,
                        cx*cz*sy - sx*cy*sz,
                        cx*cy*sz + sx*cz*sy])


    def quat2mat(self, q):
    
        ''' Calculate rotation matrix corresponding to quaternion
        https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
        Parameters
        ----------
        q : 4 element array-like
        Returns
        -------
        M : (3,3) array
        Rotation matrix corresponding to input quaternion *q*
        Notes
        -----
        Rotation matrix applies to column vectors, and is applied to the
        left of coordinate vectors.  The algorithm here allows non-unit
        quaternions.
        References
        '''
        
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-8:
            return np.eye(3)
        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array(
            [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


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

        #drive_name = KITTI360_IO.drive_foldername(cur_seq)
        drive_name = str(cur_seq).zfill(2)
        cur_lidar_dir = os.path.join(self.datapath, 'sequences', drive_name)

        # ----- load poses -----

        Tr_path = os.path.join(self.datapath, 'calib', drive_name, 'calib.txt')
        Tr_data = read_calib_file(Tr_path)
        Tr_data = Tr_data['Tr']
        Tr = Tr_data.reshape(3,4)
        Tr = np.vstack((Tr, np.array([0, 0, 0, 1.0])))

        # 'ground_truth_pose/kitti_T_diff/' + self.file_map[cur_seq] + '_diff.npy'
        poses_path = os.path.join(self.datapath, 'poses_diff', f'{str(cur_seq).zfill(2)}_diff.npy')
        pose = np.load(poses_path)

        T_diff = np.eye(4)
        for i in range(frame_gap):
            frame = cur_idx_pc2 - i
            if frame <= cur_idx_pc1:
                break
            curr_pose = np.eye(4)
            curr_pose[:3,:] = pose[frame, :].reshape((3,4))
            T_diff = curr_pose @ T_diff

        # ----- load point clouds -----

        pc1_bin = os.path.join(cur_lidar_dir, 'velodyne', str(cur_idx_pc1).zfill(6) + '.bin')
        pc2_bin = os.path.join(cur_lidar_dir, 'velodyne', str(cur_idx_pc2).zfill(6) + '.bin')

        point1 = np.fromfile(pc1_bin, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(pc2_bin, dtype=np.float32).reshape(-1, 4)

        if point1.shape[0] < point2.shape[0]:
            n = point1.shape[0]
        else:
            n = point2.shape[0] # num points of pc1 and pc2

        point1 = point1[:n, :3] # :3
        point2 = point2[:n, :3] # :3

        add = np.ones((n, 1))
        point1 = np.concatenate([point1, add], axis = -1)
        point2 = np.concatenate([point2, add], axis = -1)

        point1 = np.matmul(Tr, point1.T)
        point2 = np.matmul(Tr, point2.T)

        point1 = point1.T[ :, :3]
        point2 = point2.T[ :, :3]

        point1 = self.filter_pcd(point1)
        point2 = self.filter_pcd(point2)

        n = point1.shape[0]
        
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
          
            T_gt = np.matmul(T_diff, np.linalg.inv(T_trans))

        else:
            # transform the point cloud using only Tr_diff
            T_gt = T_diff

        R_gt = T_gt[:3, :3]
        t_gt = np.squeeze(T_gt[:3, 3])

        z_gt, y_gt, x_gt = self.mat2euler(M = R_gt)
        q_gt = self.euler2quat(z = z_gt, y = y_gt, x = x_gt)

        # q_gt = KITTI360_TOOLS.mat2quat(R_gt, scalar_last=self.config.scalar_last)

        point1_float = point1.astype(np.float32)
        point2_float = point2.astype(np.float32)
        q_gt_float = q_gt.astype(np.float32)
        t_gt_float = t_gt.astype(np.float32)

        # I switched PCD1 and PCD2
        return point2_float, point1_float, q_gt_float, t_gt_float, cur_seq, cur_idx_pc2


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

    root_path = '/run/user/1000/gvfs/sftp:host=10.186.4.18/mnt/isilon/melamine/KITTI/dataset'
    npoints = 8192
    is_training = True
    is_testing = False

    config = KittiOdometryConfig()
    config.root_dir = root_path
    config.num_points = 8192
    config.train_sequences = [3]
    config.test_sequences = [0, 3, 7, 9]
    config.scalar_last = False
    config.velo_to_pose = True
    config.augment = True
    kitti_dataset = KittiOdometryDataset(config = config, is_training = is_training, is_testing = is_testing)

    idx = np.random.choice(np.arange(len(kitti_dataset)), 1)[0]
    print(idx)
    points2, points1, q, t,_,_ = kitti_dataset[idx]

    coordinate_frame_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=3)

    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(points1)
    pcd1.colors = open3d.utility.Vector3dVector(np.tile(np.array([1., 0., 0.]), (points1.shape[0], 1)))

    # -----------

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(points2)
    pcd2.colors = open3d.utility.Vector3dVector(np.tile(np.array([0., 0., 1.]), (points2.shape[0], 1)))

    R = KITTI360_TOOLS.quat2mat(q)#, scalar_last=kitti_dataset.config.scalar_last)
    R = R.reshape(3,3)
    T = t.reshape(3)
    transformation = np.eye(4)
    transformation[:3,:3] = R
    transformation[:3,3] = T

    coordinate_frame_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    coordinate_frame_2 = coordinate_frame_2.transform(transformation)

    open3d.visualization.draw_geometries([pcd1, pcd2, coordinate_frame_1, coordinate_frame_2])
