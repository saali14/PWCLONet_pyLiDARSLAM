import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
import warnings

# Hydra and OmegaConf
from omegaconf import DictConfig


import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from slam.common.utils                          import assert_debug
from slam.models.PWCLONet.costvolume            import CostVolume
from slam.models.PWCLONet.flowpredictor         import FlowPredictor
from slam.models.PWCLONet.pose_warp_refinement  import PoseWarpRefinement
from slam.models.PWCLONet.pose_calculator       import PoseCalculator
from slam.common.pose                           import Pose


from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModulePWCLONet


class PWCLONet(nn.Module):
    """
    PWCLONet is a network regressing the 6 parameters of rigid transformation
    From a pair of images or a single image

    Inputs:
    --------
        * `xyz_f1`:       [B, 3, N1]
        * `points_f1`:    [B, C1, N1]
        * `xyz_f2`:       [B, 3, N2]
        * `points_f2`:    [B, C2, N2]

    Returns:
    --------
        * `pose_params`: [B, 4, 7] where 4=nb_levels and 7=3(translation)+4(quaternions)
        
    """

    def __init__(self, config: DictConfig, pose: Pose = Pose("quaternions")):
        nn.Module.__init__(self)

        self.config = config
        self.pose = pose
        self.num_out_poses = self.config.get("num_out_poses", 1)
        self.num_input_channels = self.config.num_input_channels
        self.sequence_len = self.config.sequence_len
        self.device = torch.device(self.config.device)

        if self.num_out_poses != 1:
            warnings.warn('current version of PWCLONet allows predicting only one pose')

        self.nb_levels = self.config.get("num_out_poses", 4)
        
        ############ Siamese Point Feature Pyramid ############
        self.psa_1 = PointnetSAModulePWCLONet(npoint=2048, nsample=32, mlp=[0, 8, 8, 16], bn=True) #, radius=0.5)   # 0
        self.psa_2 = PointnetSAModulePWCLONet(npoint=1024, nsample=32, mlp=[16, 16, 16, 32], bn=True) #, radius=1.0) # 16
        self.psa_3 = PointnetSAModulePWCLONet(npoint=256, nsample=16, mlp=[32, 32, 32, 64], bn=True) #, radius=2.0)  # 32
        self.psa_4 = PointnetSAModulePWCLONet(npoint=64, nsample=16, mlp=[64, 64, 64, 128], bn=True) #, radius=4.0)  # 64


        ############ ATTENTIVE COST VOLUME ############
        self.cost_volume = CostVolume(nsample=4, nsample_q=32, in_channel1=64, in_channel2=64, mlp1=[128, 64, 64], mlp2=[128, 64])

        # flow feature encoding layer: to make the scene flow smoother
        # This further flow encoding mixes flow features in larger receptive fields, 
        # which makes the indistinguishable objects obtain more surrounding information
        self.flow_feature_encoding = PointnetSAModulePWCLONet(npoint=64, nsample=16, mlp=[64, 128, 64, 64], bn=True) #, radius=4.0) # 64
        

        ############ Hierarchical Embedding Mask Optimization ############
        ######### LAYER 4 #########

        self.l4_flow_predictor = FlowPredictor(in_channel=128+64, mlp=[128,64])

        self.pose_calculator_4 = PoseCalculator(in_channel=64, out_channel=256, kernel_size=1, padding='valid', activation=None, squeeze=True)


        ############ Pose Warp-Refinement ############
        ######### LAYER 3 #########

        # `in_channel_f1_prev` is the output dim of `flow_feature_encoding` = 64
        # `in_channel_mask` is the output dim of `l4_flow_predictor` = 64
        self.pose_warp_refinement_3 = PoseWarpRefinement(in_channel_f1=64, in_channel_f2=64, in_channel_f1_prev=64, in_channel_mask=64, radius=2.0, last_pose_estimation=False, device=self.config.device, scalar_last=self.config.scalar_last)
        
        ######### LAYER 2 #########

        # `in_channel_f1_prev` is the output dim of the flow_predictor module inside the pose_warp_refinement module = 64
        # `in_channel_mask` is the output dim of the `setupconv_mask` module inside the pose_warp_refinement module = 64
        self.pose_warp_refinement_2 = PoseWarpRefinement(in_channel_f1=32, in_channel_f2=32, in_channel_f1_prev=64, in_channel_mask=64, radius=1.0, last_pose_estimation=False, device=self.config.device, scalar_last=self.config.scalar_last)
        
        ######### LAYER 1 #########

        # `in_channel_f1_prev` is the output dim of the flow_predictor module inside the pose_warp_refinement module = 64
        # `in_channel_mask` is the output dim of the `setupconv_mask` module inside the pose_warp_refinement module = 64
        self.pose_warp_refinement_1 = PoseWarpRefinement(in_channel_f1=16, in_channel_f2=16, in_channel_f1_prev=64, in_channel_mask=64, radius=0.5, last_pose_estimation=True, device=self.config.device, scalar_last=self.config.scalar_last)


    def forward(self, xyz_f1, points_f1, xyz_f2, points_f2, bn_decay=None):
        """
        Inputs:
        --------
            * `xyz_f1`:       [B, 3, N]
            * `points_f1`:    [B, C1, N]
            * `xyz_f2`:       [B, 3, N]
            * `points_f2`:    [B, C2, N]

        Returns:
        --------
            * `pose_params`: [B, 4, 7] where 4=nb_levels and 7=3(translation)+4(quaternions)
        """

        batch_size = xyz_f1.size(0)

        xyz_f1_t = torch.permute(xyz_f1, (0, 2, 1)).contiguous()    # [B, N, 3]
        xyz_f2_t = torch.permute(xyz_f2, (0, 2, 1)).contiguous()    # [B, N, 3]

        ############ SET ABSTRACTION ############
        
        new_xyz_f1_1_t, new_points_f1_1 = self.psa_1(xyz_f1_t, points_f1)
        new_xyz_f1_2_t, new_points_f1_2 = self.psa_2(new_xyz_f1_1_t, new_points_f1_1)
        new_xyz_f1_3_t, new_points_f1_3 = self.psa_3(new_xyz_f1_2_t, new_points_f1_2)
        new_xyz_f1_4_t, new_points_f1_4 = self.psa_4(new_xyz_f1_3_t, new_points_f1_3)

        new_xyz_f2_1_t, new_points_f2_1 = self.psa_1(xyz_f2_t, points_f2)
        new_xyz_f2_2_t, new_points_f2_2 = self.psa_2(new_xyz_f2_1_t, new_points_f2_1)
        new_xyz_f2_3_t, new_points_f2_3 = self.psa_3(new_xyz_f2_2_t, new_points_f2_2)
        new_xyz_f2_4_t, new_points_f2_4 = self.psa_4(new_xyz_f2_3_t, new_points_f2_3)

        # ----------- Level 1 --------------

        new_xyz_f1_1 = torch.permute(new_xyz_f1_1_t, (0, 2, 1)).contiguous()    # [B, N, 3]
        new_xyz_f2_1 = torch.permute(new_xyz_f2_1_t, (0, 2, 1)).contiguous()    # [B, N, 3]

        # ----------- Level 2 --------------        

        new_xyz_f1_2 = torch.permute(new_xyz_f1_2_t, (0, 2, 1)).contiguous()
        new_xyz_f2_2 = torch.permute(new_xyz_f2_2_t, (0, 2, 1)).contiguous()

        # ----------- Level 3 --------------

        new_xyz_f1_3 = torch.permute(new_xyz_f1_3_t, (0, 2, 1)).contiguous()
        new_xyz_f2_3 = torch.permute(new_xyz_f2_3_t, (0, 2, 1)).contiguous()

        # ----------- Level 4 --------------

        new_xyz_f1_4 = torch.permute(new_xyz_f1_4_t, (0, 2, 1)).contiguous()
        new_xyz_f2_4 = torch.permute(new_xyz_f2_4_t, (0, 2, 1)).contiguous()

        ############ ATTENTIVE COST VOLUME ############

        flow_embedding = self.cost_volume(new_xyz_f1_3, new_points_f1_3, new_xyz_f2_3, new_points_f2_3)

        xyz_f1_4_t, embedding_features_4 = self.flow_feature_encoding(new_xyz_f1_3_t, flow_embedding)

        new_xyz_f1_4_t = torch.clone(xyz_f1_4_t)
        new_xyz_f1_4 = torch.permute(xyz_f1_4_t, (0, 2, 1)).contiguous()

        ############ Hierarchical Embedding Mask Optimization ############

        embedding_mask_4 = self.l4_flow_predictor(new_points_f1_4, embedding_features_4)
        W_cost_volume_4 = F.softmax(embedding_mask_4, dim=2) # [B, self.l3_flow_predictor.out_channel, N1_(l+1)]

        q_4, t_4 = self.pose_calculator_4(embedding_features_4, W_cost_volume_4)

        assert len(q_4.size()) == len(t_4.size()) == 2, f'[PWCLONet] Wrong shape q={q_4.size()} and t={t_4.size()}'

        ############ Pose Warp-Refinement ############

        q_3, t_3, embedding_features_3, embedding_mask_3 = self.pose_warp_refinement_3(new_xyz_f1_3, new_points_f1_3, new_xyz_f2_3, new_points_f2_3, new_xyz_f1_4, embedding_features_4, embedding_mask_4, q_4, t_4)
        q_2, t_2, embedding_features_2, embedding_mask_2 = self.pose_warp_refinement_2(new_xyz_f1_2, new_points_f1_2, new_xyz_f2_2, new_points_f2_2, new_xyz_f1_3, embedding_features_3, embedding_mask_3, q_3, t_3)
        q_1, t_1, embedding_features_1, embedding_mask_1 = self.pose_warp_refinement_1(new_xyz_f1_1, new_points_f1_1, new_xyz_f2_1, new_points_f2_1, new_xyz_f1_2, embedding_features_2, embedding_mask_2, q_2, t_2)

        #embedding_mask_log_3 = torch.mean(torch.permute(F.softmax(embedding_mask_3.detach().cpu(), dim=2), (0, 2, 1)), dim=-1)  # B, N
        #embedding_mask_log_2 = torch.mean(torch.permute(F.softmax(embedding_mask_2.detach().cpu(), dim=2), (0, 2, 1)), dim=-1)  # B, N
        embedding_mask_log_1 = torch.linalg.norm(torch.permute(F.softmax(embedding_mask_1.detach().cpu(), dim=2), (0, 2, 1)), dim=-1, ord=2)  # B, N

        log_dict = {
            'embedding_mask': embedding_mask_log_1,
            'point_cloud': new_xyz_f1_1_t.detach().cpu(),
            #'embedding_mask_2': embedding_mask_log_2,
            #'embedding_mask_3': embedding_mask_log_3,
        }

        q_norm_1 = q_1 / (torch.sqrt(torch.sum(q_1 * q_1, dim=-1, keepdim=True) + 1e-10) + 1e-10)   # [B, 4]
        q_norm_2 = q_2 / (torch.sqrt(torch.sum(q_2 * q_2, dim=-1, keepdim=True) + 1e-10) + 1e-10)   # [B, 4]
        q_norm_3 = q_3 / (torch.sqrt(torch.sum(q_3 * q_3, dim=-1, keepdim=True) + 1e-10) + 1e-10)   # [B, 4]
        q_norm_4 = q_4 / (torch.sqrt(torch.sum(q_4 * q_4, dim=-1, keepdim=True) + 1e-10) + 1e-10)   # [B, 4]

        pose_params_1 = torch.unsqueeze(torch.cat((t_1, q_norm_1), dim=-1).reshape(-1, 7), 1)   # [B, 1, 7]
        pose_params_2 = torch.unsqueeze(torch.cat((t_2, q_norm_2), dim=-1).reshape(-1, 7), 1)   # [B, 1, 7]
        pose_params_3 = torch.unsqueeze(torch.cat((t_3, q_norm_3), dim=-1).reshape(-1, 7), 1)   # [B, 1, 7]
        pose_params_4 = torch.unsqueeze(torch.cat((t_4, q_norm_4), dim=-1).reshape(-1, 7), 1)   # [B, 1, 7]

        pose_params = torch.cat((pose_params_1, pose_params_2, pose_params_3, pose_params_4), dim=1)
        
        return pose_params, log_dict


class PWCLONET(Enum):
    pwclonet = PWCLONet

    @staticmethod
    def load(config: DictConfig, pose: Pose = Pose("quaternions")):
        assert_debug("type" in config)
        assert_debug(config.type in PWCLONET.__members__)

        return PWCLONET.__members__[config.type].value(config)