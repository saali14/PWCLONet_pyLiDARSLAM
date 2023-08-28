
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)


import slam.models.PWCLONet.PWCLO_utils            as pwclo
from slam.models.PWCLONet.costvolume               import CostVolume
from slam.models.PWCLONet.flowpredictor            import FlowPredictor
from slam.models.PWCLONet.pose_calculator          import PoseCalculator

from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModulePWCLONet

from slam.common.pose import Pose


class PoseWarpRefinement(nn.Module):
    """
        To achieve the coarse-to-ﬁne reﬁnement process in an end-to-end fashion, 
        we propose the differentiable warp-reﬁnement module based on pose transformation
        This module contains several key parts:
            * `set up-conv layer`, `pose warping`, `embedding feature and embedding mask reﬁnement`, and `pose reﬁnement`\n

        Inputs:
            * xyz_f1, xyz_f2, xyz_f1_prev:          [B, 3, N]
            * points_f1, points_f2, points_f1_prev: [B, C1/C2/C3, N]
            * embedding_mask_prev:                  [B, C, N']
            * q_prev and t_prev:                    [B, 4] and [B,3]\n
        Return:
            * q and t:                              [B, 4] and [B,3]
    """

    def __init__(self, in_channel_f1: int, in_channel_f2: int, in_channel_f1_prev: int, in_channel_mask: int,
                 knn:bool =False, radius: float=0.0, last_pose_estimation: bool = False, 
                 pose: Pose = Pose("quaternions"), device: str = "cpu", scalar_last: bool = True):
        super(PoseWarpRefinement, self).__init__()

        if (not knn) and (radius == 0.0):
            raise RuntimeError('PoseWarpRefinement: when `knn` is set to False, `radius` should be precised.\n\
                               Use radius value used in the Set Abstraction phase of the corresponding layer')
        
        self.pose = pose
        self.device = device
        self.scalar_last = scalar_last
        
        self.last_pose_estimation = last_pose_estimation

        self.in_channel = [in_channel_f1, in_channel_f2, in_channel_f1_prev]

        # ------- Feature Propagation -------
        #self.setupconv_features = SetUpConv(nsample=8, f1_channel=in_channel_f1, f2_channel=in_channel_f1_prev, mlp=[128,64], mlp2=[64], knn=knn, radius=radius*0.2)
        #self.setupconv_mask = SetUpConv(nsample=8, f1_channel=in_channel_f1, f2_channel=in_channel_mask, mlp=[128,64], mlp2=[64], knn=knn, radius=radius*0.2)

        # if knn is True, radius is ignored 
        self.setupconv_features = PointnetFPModulePWCLONet(nsample=8, mlp=[in_channel_f1_prev,128,64], post_mlp=[64+in_channel_f1,64], radius=radius*0.2, knn=True, use_xyz=True, bn=True)
        self.setupconv_mask = PointnetFPModulePWCLONet(nsample=8, mlp=[in_channel_mask,128,64], post_mlp=[64+in_channel_f1,64], radius=radius*0.2, knn=True, use_xyz=True, bn=True)

        # ------- Re-embedding Features -------
        self.cost_volume = CostVolume(nsample=4, nsample_q=6, in_channel1=in_channel_f1, in_channel2=in_channel_f2, mlp1=[128,64,64], mlp2=[128,64])

        # ------- Embedding Features at the l-th level -------
        self.flow_predictor_features = FlowPredictor(in_channel=in_channel_f1+64+64, mlp=[128,64])

        # ------- Embedding Mask Refinement at the l-th level -------
        if not self.last_pose_estimation:
            self.flow_predictor_mask = FlowPredictor(in_channel=in_channel_f1+64+64, mlp=[128,64])

        # ------- Pose Refinement -------
        self.pose_calculator = PoseCalculator(in_channel=64, out_channel=256, kernel_size=1, padding='valid', activation=None, pose=self.pose, squeeze=False)

        self.out_channel = [self.pose.num_rot_params(), 3, 64]

        
    def forward(self, xyz_f1, points_f1, xyz_f2, points_f2, xyz_f1_prev, points_f1_prev, embedding_mask_prev, q_prev, t_prev):
        """
        Inputs:
            * xyz_f1, xyz_f2, xyz_f1_prev:          [B, 3, N1/N2/N3]
            * points_f1, points_f2, points_f1_prev: [B, C1/C2/C3, N1/N2/N3]
            * embedding_mask_prev:                  [B, C, N3]
            * q_prev and t_prev:                    [B, 4] and [B,3]\n
        Return:
            * q and t:                              [B, 4] and [B,3]
        """
        # -1_C, -1_N: means certain number of channels or points

        batch_size = xyz_f1.size(0)

        q_coarse = torch.reshape(q_prev, [batch_size, 4, 1]) # 4 instead of self.pose.num_rot_params()
        t_coarse = torch.reshape(t_prev, [batch_size, 3, 1])

        xyz_f1_t = torch.permute(xyz_f1, (0, 2, 1)).contiguous()
        xyz_f1_prev_t = torch.permute(xyz_f1_prev, (0, 2, 1)).contiguous()

        # ------- Feature Propagation -------
        coarse_embedding_features = self.setupconv_features(xyz_f1_t, xyz_f1_prev_t, points_f1, points_f1_prev) # [B, -1_C, N1]
        coarse_embedding_masks = self.setupconv_mask(xyz_f1_t, xyz_f1_prev_t, points_f1, embedding_mask_prev) # [B, -1_C, N1]

        # ------- Pose Warping -------
        warped_xyz_f1 = pwclo.warp(xyz_f1, q_coarse, t_coarse, self.device, self.scalar_last) # [B, 3, N]

        # ------- Re-embedding Features -------
        residual_embedding = self.cost_volume(warped_xyz_f1, points_f1, xyz_f2, points_f2) # [B, -1_C, N]     

        # ------- Embedding Features at the l-th level -------
        embedding_features = self.flow_predictor_features(points_f1, residual_embedding, coarse_embedding_features) # [B, -1_C, -1_N]

        # ------- Embedding Mask Refinement at the l-th level -------
        if not self.last_pose_estimation:
            embedding_mask = self.flow_predictor_mask(coarse_embedding_masks, embedding_features, points_f1) # [B, -1_C, -1_N]
        else:
            embedding_mask = coarse_embedding_masks # [B, -1_C, -1_N]
        W_cost_volume =  F.softmax(embedding_mask, dim=2) # [B, -1_C, -1_N]

        # ------- Pose Refinement -------
        q_det, t_det = self.pose_calculator(embedding_features, W_cost_volume) # [B, 4, 1], [B, 3, 1]

        assert len(q_det.size()) == len(t_det.size()) == 3, f'[Pose Warp Refinement] Wrong shape q={q_det.size()} and t={t_det.size()}'

        ####### ///!\\\ in the paper they used `q_det` but in the code the used `q_coarse`
        # I already changed it
        """
        q_det_ = torch.squeeze(q_det, dim=1)
        q_det_inv = pwclo.inv_q(q_det_)

        t_coarse_trans = torch.cat((torch.zeros([batch_size, 1, 1]), t_coarse), dim=-1)
        t_coarse_trans = pwclo.mul_q_point(q_det, t_coarse_trans)
        t_coarse_trans = pwclo.mul_point_q(t_coarse_trans, q_det_inv)
        t_coarse_trans = t_coarse_trans[:, :, 1:]
        t = torch.squeeze(t_coarse_trans + t_det)
        """
        q = torch.squeeze(pwclo.mul_point_q(q_det, q_coarse, self.scalar_last), dim=2) # [B, 4]

        #####################

        # My suggestion
        # t = torch.add(torch.squeeze(t_coarse, dim=2), torch.squeeze(t_det, dim=2))
        #####################

        # The code given by PWCLONET paper
        t = torch.squeeze(pwclo.warp(t_coarse, q_det, t_det, self.device, self.scalar_last), dim=2) # [B, 3]

        # The original code
        # t = torch.squeeze(pwclo.warp(t_coarse, q_coarse, t_det, self.device, self.scalar_last), dim=2) # [B, 3]

        assert len(q.size()) == len(t.size()) == 2, f'[Pose Warp Refinement] Wrong shape q={q.size()} and t={t.size()}'
        assert q.size(0) == t.size(0) == batch_size, f'[Pose Warp Refinement] Wrong shape q={q.size()} and t={t.size()}'
        assert q.size(1) == 4, f'[Pose Warp Refinement] Wrong shape q: {q.size()}'
        assert t.size(1) == 3, f'[Pose Warp Refinement] Wrong shape t: {t.size()}'

        return q, t, embedding_features, embedding_mask

