
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)


from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pytorch_utils import Conv1d
from slam.common.pose import Pose


class PoseCalculator(nn.Module):
    """
        Calculate pose using embedding features and weighting mask
        Inputs:
            * embedding_features:   [B, C, N]
            * mask:                 [B, C, N]\n
        Return:
            * q:                    [B, 4] if squeeze else [B, 1, 4]
            * t:                    [B, 3] if squeeze else [B, 1, 3]
    """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 1, padding = 'valid', activation = None, pose: Pose = Pose("quaternions"), squeeze: bool = True, bn_decay=None):
        super(PoseCalculator, self).__init__()

        self.pose = pose
        self.squeeze = squeeze

        self.conv1d_q_t = Conv1d(in_size=in_channel, out_size=out_channel, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)
        self.conv1d_q = Conv1d(in_size=out_channel, out_size=4, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)    # 4 instead of self.pose.num_rot_params()
        self.conv1d_t = Conv1d(in_size=out_channel, out_size=3, kernel_size=kernel_size, padding=padding, activation=activation, init=torch.nn.init.xavier_uniform_)

        # to be checked later
        # Initialize Scale to allow stable training
        """torch.nn.init.xavier_uniform_(self.conv1d_q.weight, 0.01)
        torch.nn.init.xavier_uniform_(self.conv1d_t.weight, 0.01)"""

        
    def forward(self, embedding_features, mask):
        """
        Inputs:
            * embedding_features:   [B, C, N]
            * mask:                 [B, C, N]\n
        Return:
            * q:                    [B, 4] if squeeze else [B, 4, 1]
            * t:                    [B, 3] if squeeze else [B, 3, 1]
        """

        # cost_volume_sum: [B, C, 1]
        cost_volume_sum = torch.sum(embedding_features * mask, dim=2, keepdim=True)
        # cost_volume_sum_big: [B, self.conv1d_q_t.out_channel, 1]
        cost_volume_sum_big = self.conv1d_q_t(cost_volume_sum)

        # cost_volume_sum_q: [B, self.conv1d_q_t.out_channel, 1]
        cost_volume_sum_q = F.dropout(cost_volume_sum_big, p=0.5, training=self.training)
        # cost_volume_sum_t: [B, self.conv1d_q_t.out_channel, 1]
        cost_volume_sum_t = F.dropout(cost_volume_sum_big, p=0.5, training=self.training)

        # q: [B, 4, 1]
        q = self.conv1d_q(cost_volume_sum_q)
        q = q / (torch.sqrt(torch.sum(q*q, dim=1, keepdim=True) + 1e-10) + 1e-10)
        
        # t: [B, 3, 1]
        t = self.conv1d_t(cost_volume_sum_t)

        assert len(q.size()) == len(t.size()) == 3, f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'
        assert q.size(0) == t.size(0) == embedding_features.size(0), f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'
        assert q.size(1) == 4, f'[Pose Calculator] Wrong shape q: {q.size()}'
        assert t.size(1) == 3, f'[Pose Calculator] Wrong shape t: {t.size()}'
        assert q.size(2) == t.size(2) == 1, f'[Pose Calculator] Wrong shape q={q.size()} and t={t.size()}'

        if self.squeeze:
            # q: [B, 4]
            q = torch.squeeze(q, dim=2)
            # t: [B, 3]
            t = torch.squeeze(t, dim=2)
        
        return q, t

