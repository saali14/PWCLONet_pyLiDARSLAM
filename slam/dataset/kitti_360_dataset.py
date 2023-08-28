import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R, Slerp
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
import torch

# Hydra and OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from pyLiDAR_SLAM.slam.common.geometry      import estimate_timestamps
from pyLiDAR_SLAM.slam.common.projection    import SphericalProjector
from pyLiDAR_SLAM.slam.common.utils         import assert_debug
from pyLiDAR_SLAM.slam.dataset.configuration import DatasetLoader, DatasetConfig
from pyLiDAR_SLAM.slam.dataset.kitti_dataset import kitti_read_scan, KITTIOdometrySequence
from pyLiDAR_SLAM.slam.eval.eval_odometry   import compute_relative_poses

from kitti360Scripts.kitti360scripts.helpers.common import KITTI360_IO


class KITTI360Sequence(Dataset):
    """
    Dataset for a Sequence of KITTI-360 lidar dataset

    Attributes:
        kitti360_root_dir (str): The path to KITTI-360 data
        drive_id (int): The name id of drive [0, 2, 3, 4, 5, 6, 7, 9, 10]

    __get_item__(idx):
        returns a data_dict of the `idx`th frame in the sequence `drive_id` that contains:
            * `numpy_pc`: 3D point cloud of the frame in the lidar coordinates
            * `numpy_reflectance`: the corresponding reflectance values of each point in the the 3D PCD
            * `numpy_pc_timestamps`: the corresponding timestamps of each point in the 3D PCD
            * `vertex_map` [if set in `corrected_lidar_channel`]: the range image of the frame
            * `absolute_pose_gt`: the relative poses between the system coordinates of the first frame and the current frame
    """

    __sequence_size = {
        0: 11518,
        2: 19240,
        3: 1031,
        4: 11587,
        5: 6743,
        6: 9699,
        7: 3396,
        9: 14056,
        10: 3836
    }

    # -- ME -- added `corrected_lidar_channel` and `lidar_projector`
    def __init__(self,
                 kitti360_root_dir: str,
                 drive_id: int,
                 corrected_lidar_channel: str = None,
                 lidar_projector: SphericalProjector = None):
        
        self.root_dir: Path = Path(kitti360_root_dir)

        # -- ME --
        self.corrected_lidar_channel = corrected_lidar_channel
        self.lidar_projector = lidar_projector

        assert_debug(drive_id in [0, 2, 3, 4, 5, 6, 7, 9, 10])
        sequence_foldername = KITTI360_IO.drive_foldername(drive_id)
        velodyne_path = self.root_dir / "data_3d_raw" / sequence_foldername / "velodyne_points"
        self.lidar_path = velodyne_path / "data"

        assert_debug(self.lidar_path.exists(), f"The drive directory {self.lidar_path} does not exist")
        self.size: int = self.__sequence_size[drive_id]
        self.gt_poses = KITTI360_IO.get_sequence_poses(kitti360_root_dir, drive_id)
        # -- ME -- first frame of lidar to current frame of lidar
        self.gt_poses = np.einsum("ij,njk->nik", np.linalg.inv(self.gt_poses[0]), self.gt_poses)
        

    def __len__(self):
        return self.size
    

    @staticmethod
    def _correct_scan(scan: np.ndarray):
        """
        Corrects the calibration of KITTI's HDL-64 scan
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)

        return corrected_scan

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            A dictionary with the mapping defined in the constructor
        """
        assert_debug(idx < self.size)
        data_dict = {}

        # -- ME -- seq 2 doesn't have frames starting from 0
        #import os
        #bin_files = os.listdir(str(self.lidar_path))
        #sorted_bin_files = sorted(bin_files)
        #xyz_r = kitti_read_scan(str(self.lidar_path / sorted_bin_files[idx]))
         

        xyz_r = kitti_read_scan(str(self.lidar_path / f"{idx:010}.bin"))
        data_dict[DatasetLoader.numpy_pc_key()] = KITTIOdometrySequence.correct_scan(xyz_r[:, :3])
        data_dict["numpy_pc_reflectance"] = xyz_r[:, 3:]
        data_dict["numpy_pc_timestamps"] = estimate_timestamps(xyz_r[:, :3], phi_0=np.pi, clockwise=True)

        # -- ME -- add `vertex_map` when `projector` is specified
        if self.corrected_lidar_channel is not None:
            scan = torch.from_numpy(data_dict[DatasetLoader.numpy_pc_key()][:, :3]).unsqueeze(0)
            data_dict[self.corrected_lidar_channel] = self.lidar_projector.build_projection_map(scan)[0]

        if self.gt_poses is not None:
            data_dict[DatasetLoader.absolute_gt_key()] = self.gt_poses[idx]
            if idx == (len(self.gt_poses) - 1):
                data_dict[DatasetLoader.relative_gt_key()] = np.eye(4)
            else:
                data_dict[DatasetLoader.relative_gt_key()] = np.linalg.inv(self.gt_poses[idx]) @ self.gt_poses[idx+1]

        return data_dict


# ----------------------------------------------------------------------------------------------------------------------
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
    train_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7, 9, 10])
    test_sequences: list = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7])
    eval_sequences: list = field(default_factory=lambda: [9, 10])

    # -- ME --
    lidar_key: str = DatasetLoader.vertex_map_key()
    num_points: int = -1


# Hydra -- stores a KITTIConfig `kitti_360` in the `dataset` group
cs = ConfigStore.instance()
cs.store(group="dataset", name="kitti_360", node=KITTI360Config)


# ----------------------------------------------------------------------------------------------------------------------
class KITTI360DatasetLoader(DatasetLoader):
    """
    Configuration for KITTI-360 dataset
    see http://www.cvlibs.net/datasets/kitti-360/
    """

    def __init__(self, config: KITTI360Config):
        super().__init__(config)
        self.root_dir = Path(self.config.root_dir)
        assert_debug(self.root_dir.exists())

    def projector(self) -> SphericalProjector:
        """Default SphericalProjetor for KITTI (projection of a pointcloud into a Vertex Map)"""
        assert isinstance(self.config, KITTI360Config)
        lidar_height = self.config.lidar_height
        lidar_with = self.config.lidar_width
        up_fov = self.config.up_fov
        down_fov = self.config.down_fov
        # Vertex map projector
        projector = SphericalProjector(lidar_height, lidar_with, 3, up_fov, down_fov)
        return projector

    def get_ground_truth(self, drive_id: str):
        """Returns the relative ground truth poses associated to a sequence of KITTI-360"""
        drive_id = int(drive_id)
        gt_poses = KITTI360_IO.get_sequence_poses(self.root_dir, drive_id)
        gt_poses = np.einsum("ij,njk->nik", np.linalg.inv(gt_poses[0]), gt_poses)
        return compute_relative_poses(gt_poses)

    # -- ME -- added `sequences`
    def sequences(self, sequences:dict=None):
        """
        Returns
        -------
        (train_dataset, eval_dataset, test_dataset, transform) : tuple
        train_dataset : (list, list)
            A list of dataset_config (one for each sequence of KITTI's Dataset),
            And the list of sequences used to build them
        eval_dataset : (list, list)
            idem
        test_dataset : (list, list)
            idem
        transform : callable
            A transform to be applied on the dataset_config
        """

        # -- ME --
        if self.config.with_vertex_map:
            lidar_channel = self.config.lidar_key
        else:
            lidar_channel = None
        
        # Sets the path of the kitti benchmark
        if not sequences is None:
            train_sequence_ids = [str(_id) for _id in sequences['train']]
            eval_sequence_ids = [str(_id) for _id in sequences['eval']]
            test_sequence_ids = [str(_id) for _id in sequences['test']]
        else:
            train_sequence_ids = [str(_id) for _id in self.config.train_sequences]
            eval_sequence_ids = [str(_id) for _id in self.config.eval_sequences]
            test_sequence_ids = [str(_id) for _id in self.config.test_sequences]

        def __get_datasets(sequences: list):
            if sequences is None or len(sequences) == 0:
                return None

            # -- ME -- added `lidar_channel` and `self.projector()`
            datasets = [KITTI360Sequence(str(self.root_dir),
                                         int(sequence_id),
                                         lidar_channel,
                                         self.projector()) for sequence_id in sequences]
            return datasets

        train_datasets = __get_datasets(train_sequence_ids)
        eval_datasets = __get_datasets(eval_sequence_ids)
        test_datasets = __get_datasets(test_sequence_ids)

        return (train_datasets, train_sequence_ids), \
               (eval_datasets, eval_sequence_ids), \
               (test_datasets, test_sequence_ids), lambda x: x
