
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pytorch_utils as pt_utils

class FlowPredictor(nn.Module):
    """
        Generates trainable embedding mask for prioritizing embedding features of N points in PC1
        in order to generate pose transformation from embedding features 
        M = softmax(sharedMLP(embedding features, features of PC1))\n

        Inputs:
            * `points_f1`:      [B, C1, N]
            * `cost_volume`:    [B, C2, N]
            * `upsampled_feat`: [B, C', N] (default: `None`)\n
        Returns:
            * `points_concat`:  [B, mlp[-1], N]
    """

    def __init__(self, in_channel, mlp, bn_decay=None):
        super(FlowPredictor, self).__init__()

        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()

        self.in_channel = [in_channel]

        mlp_spec = [in_channel] + mlp
        self.mlp_convs = pt_utils.SharedMLP(mlp_spec, bn=True, init=torch.nn.init.xavier_uniform_)

        self.out_channel = mlp_spec[-1]

        """
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.out_channel = last_channel
        """


    def forward(self, points_f1, cost_volume, upsampled_feat=None):
        """
        Inputs:
            * `points_f1`:      [B, C1, N]
            * `cost_volume`:    [B, C2, N]
            * `upsampled_feat`: [B, C', N] (default: `None`)\n
        Returns:
            * `points_concat`:  [B, mlp[-1], N]
        """
        if points_f1 is None:
            points_concat = cost_volume

        elif upsampled_feat != None:
            points_concat = torch.cat((points_f1, cost_volume, upsampled_feat), dim=1) # B, C1 + C2 + C', N

        elif upsampled_feat == None:
            points_concat = torch.cat((points_f1, cost_volume), dim=1) # B, C1 + C2, N

        # [B, C1 + C2 + (C'|0), N, 1]
        points_concat = torch.unsqueeze(points_concat, 3)

        points_concat = self.mlp_convs(points_concat)
        """
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points_concat = F.relu(bn(conv(points_concat)))
        """
                                        
        # [B, mlp[-1], N]
        points_concat = torch.squeeze(points_concat, dim=3)
      
        return points_concat