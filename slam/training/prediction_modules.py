from typing import Optional, Dict, Any, Union, List

from hydra.core.config_store import ConfigStore
from torch import nn as nn
import torch
import numpy as np

# Hydra and OmegaConf
from omegaconf import OmegaConf
from hydra.conf import dataclass, MISSING


import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from slam.common.pose                  import Pose
from slam.models.posenet               import POSENET
from slam.dataset.configuration        import DatasetLoader
from slam.models.PWCLONet.pwclo_net import PWCLONET


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PredictionConfig:
    """PoseNet Prediction Config"""
    num_input_channels: int = MISSING
    sequence_len: int = MISSING
    posenet_config: Optional[Dict[str, Any]] = None
    num_points: Optional[int] = -1
    device: str = "cpu"


# Hydra -- Create a group for the Prediction Config
cs = ConfigStore.instance()
cs.store(group="training/prediction", name="posenet", node=PredictionConfig)


# ----------------------------------------------------------------------------------------------------------------------
# POSENET PREDICTION MODULE
class _PoseNetPredictionModule(nn.Module):
    """
    Posenet Module
    """

    def __init__(self,
                 config: PredictionConfig,
                 pose: Pose):
        nn.Module.__init__(self)
        self.config = PredictionConfig(**config)
        self.pose = pose

        self.device = torch.device(self.config.device)

        self.num_input_channels = self.config.num_input_channels
        self.sequence_len: int = self.config.sequence_len

        config.posenet_config["sequence_len"] = self.sequence_len
        config.posenet_config["num_input_channels"] = self.num_input_channels

        self.posenet: nn.Module = POSENET.load(OmegaConf.create(config.posenet_config), pose=self.pose)

    def forward(self, data_dict: dict):
        # -- ME -- vertex_map is a range image of shape (B, seq_len, C, H, W)
        vertex_map = data_dict[DatasetLoader.vertex_map_key()]
        # -- ME -- posenet(vertex_map) returns pose params (B, num_out_poses, nb_params)
        pose_params = self.posenet(vertex_map)[:, 0] # -- ME -- we choose only the first pose output
        data_dict["pose_params"] = pose_params
        data_dict["pose_matrix"] = self.pose.build_pose_matrix(pose_params)

        if DatasetLoader.absolute_gt_key() in data_dict:
            gt = data_dict[DatasetLoader.absolute_gt_key()]
            relative_gt = gt[:, 0].inverse() @ gt[:, 1]
            data_dict["ground_truth"] = relative_gt

        return data_dict
    

# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PWCLONetPredictionConfig(PredictionConfig):
    """PWCLONet Prediction Config"""
    num_points: int = 8192
    nb_levels: int = 4
    scalar_last: bool = MISSING

# Hydra -- Create a group for the PWCLONetPrediction Config
cs = ConfigStore.instance()
cs.store(group="training/prediction", name="pwclonet", node=PWCLONetPredictionConfig)


# ----------------------------------------------------------------------------------------------------------------------
# PWCLONet PREDICTION MODULE
class _PWCLONetPredictionModule(nn.Module):
    """
    # PWCLONet Module
    """

    def __init__(self,
                 config: PWCLONetPredictionConfig,
                 pose: Pose = Pose("quaternions")):
        nn.Module.__init__(self)

        self.config = PWCLONetPredictionConfig(**config)
        self.pose = pose

        self.device = torch.device(self.config.device)

        self.num_input_channels = self.config.num_input_channels
        self.sequence_len: int = self.config.sequence_len
        assert (self.sequence_len == 2), "PWCLONet is developed to only accept 2 frames"

        self.num_points = config.num_points
        self.nb_levels = config.nb_levels
        config.posenet_config["sequence_len"] = self.sequence_len
        config.posenet_config["num_input_channels"] = self.num_input_channels
        config.posenet_config["num_points"] = self.num_points
        config.posenet_config["nb_levels"] = self.nb_levels
        config.posenet_config["device"] = self.config.device
        config.posenet_config["scalar_last"] = self.config.scalar_last

        self.pwclonet: nn.Module = PWCLONET.load(OmegaConf.create(config.posenet_config), pose=self.pose)


    def forward(self, data: Union[dict, List], bn_decay=None):
        

        if isinstance(data, dict):
            pcd_list = []
            for i in range(self.sequence_len):
                key = f'{DatasetLoader.numpy_pc_key()}_{i}'
                if key in data.keys():
                    pcd_list.append(data[key])
                else:
                    raise RuntimeError(f'key `{key}` not found in data when running the prediction module')
        
            l0_xyz_f1    = pcd_list[0][:, :self.num_points, :3] # self.num_input_channels = 3
            l0_points_f1 = pcd_list[0][:, :self.num_points, 3:] if pcd_list[0].size(-1) > 3 else None

            l0_xyz_f2    = pcd_list[1][:, :self.num_points, :3]
            l0_points_f2 = pcd_list[1][:, :self.num_points, 3:] if pcd_list[1].size(-1) > 3 else None

        elif isinstance(data, List):
            l0_xyz_f1    = data[0][:, :self.num_points, :3]
            l0_points_f1 = data[0][:, :self.num_points, 3:] if data[0].size(-1) > 3 else None

            l0_xyz_f2    = data[1][:, :self.num_points, :3]
            l0_points_f2 = data[1][:, :self.num_points, 3:] if data[1].size(-1) > 3 else None

        else:
            raise RuntimeError('Input data should be either dict or list')


        l0_xyz_f1    = torch.permute(l0_xyz_f1, (0, 2, 1)).contiguous()     # B, 3, num_points
        l0_points_f1 = torch.permute(l0_points_f1, (0, 2, 1)).contiguous() if l0_points_f1 is not None else None # B, C, num_points
        l0_xyz_f2    = torch.permute(l0_xyz_f2, (0, 2, 1)).contiguous()     # B, 3, num_points
        l0_points_f2 = torch.permute(l0_points_f2, (0, 2, 1)).contiguous() if l0_points_f2 is not None else None # B, C, num_points

        pose_params, log_dict = self.pwclonet(l0_xyz_f1, l0_points_f1, l0_xyz_f2, l0_points_f2, bn_decay=bn_decay)

        return pose_params, log_dict

        data["pose_params"] = pose_params
        #data["pose_params_1"] = pose_params_1
        #data["pose_params_2"] = pose_params_2
        #data["pose_params_3"] = pose_params_3
        #data["pose_matrix"] = self.pose.build_pose_matrix(pose_params[-1])

        if DatasetLoader.absolute_gt_key() in data:
            gt = data[DatasetLoader.absolute_gt_key()] # B, 4. 4
            relative_gt = torch.inverse(gt[:, 0]) @ gt[:, 1]
            data["ground_truth"] = relative_gt

        return data
    