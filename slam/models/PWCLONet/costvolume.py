
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils as pointutils
from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pytorch_utils as pt_utils


class CostVolume(nn.Module):
    """
        Cost Volume module or Attentive Point Mixture (Double Attentive Embedding layer for point mixture)
        Input:
            * `warped_xyz`:       (B, 3, S)
            * `warped_points`:    (B, C, S)
            * `f2_xyz`:           (B, 3, N)
            * `f2_points`:        (B, C, N)\n
        Returns:
            * `pc_feat1_new`:     (B, mlp[-1], S)\n
        /!\ condition:
            * mlp1[-1] == mlp2[-1] == mlp[-1]

    """
    def __init__(self, nsample, nsample_q, in_channel1, in_channel2, mlp1, mlp2):
        super(CostVolume, self).__init__()

        self.nsample = nsample
        self.nsample_q = nsample_q
        
        self.in_channel = [in_channel1, in_channel2, 10]
        last_channel = in_channel1 + in_channel2 + 10

        mlp1_spec = [in_channel1 + in_channel2 + 10] + mlp1
        self.mlp_convs = pt_utils.SharedMLP(mlp1_spec, bn=True, init=torch.nn.init.xavier_uniform_)

        mlp_spec_xyz_1 = [10, mlp1[-1]]
        self.mlp_conv_xyz_1 = pt_utils.SharedMLP(mlp_spec_xyz_1, bn=True, init=torch.nn.init.xavier_uniform_)

        mlp_spec_xyz_2 = [10, mlp1[-1]]
        self.mlp_conv_xyz_2 = pt_utils.SharedMLP(mlp_spec_xyz_2, bn=True, init=torch.nn.init.xavier_uniform_)

        # concatenating 3D Euclidean space encoding and first flow embeddings
        last_channel2 = mlp1_spec[-1] * 2 
        mlp2_spec = [last_channel2] + mlp2
        self.mlp2_convs = pt_utils.SharedMLP(mlp2_spec, bn=True, init=torch.nn.init.xavier_uniform_)

        last_channel3 = mlp1_spec[-1] * 2 + in_channel1   
        mlp3_spec = [last_channel3] + mlp2
        self.mlp3_convs = pt_utils.SharedMLP(mlp3_spec, bn=True, init=torch.nn.init.xavier_uniform_)

        self.out_channel = mlp3_spec[-1]


    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points):
        """
        Input:
            * warped_xyz:       (B, 3, S)
            * warped_points:    (B, C, S)
            * f2_xyz:           (B, 3, N)
            * f2_points:        (B, C, N)\n
        Returns:
            * pc_feat1_new:     (B, mlp[-1], S)
        """

        warped_xyz_t = warped_xyz.permute(0, 2, 1).contiguous()
        f2_xyz_t = f2_xyz.permute(0, 2, 1).contiguous()
        
        ### -----------------------------------------------------------
        ### FIRST AGGREGATE

        # f2_xyz_t: (B, N, 3)
        # warped_xyz_t: (B, S, 3)
        # idx_q: (B, S, k)

        _, idx_q = pt_utils.knn_point(self.nsample_q, f2_xyz_t, warped_xyz_t)

        # qi_xyz_grouped: (B, 3, S, k)
        # -- ME --
        qi_xyz_grouped = pointutils.grouping_operation(f2_xyz, idx_q)
        # qi_points_grouped: (B, C2, S, k)
        qi_points_grouped = pointutils.grouping_operation(f2_points, idx_q)

        # torch.unsqueeze(warped_xyz, 3): (B, 3, S, 1)
        # pi_xyz_expanded: (B, 3, S, k)
        pi_xyz_expanded = torch.tile(torch.unsqueeze(warped_xyz, 3), [1, 1, 1, self.nsample_q])
        # pi_points_expanded: (B, C1, S, k)
        pi_points_expanded = torch.tile(torch.unsqueeze(warped_points, 3), [1, 1, 1, self.nsample_q])
        
        # pi_xyz_diff: (B, 3, S, k)
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded
        
        # pi_euc_diff: (B, 1, S, k)
        pi_euc_diff = torch.sqrt(torch.sum(torch.square(pi_xyz_diff), dim=1 , keepdim=True) + 1e-20 )
    
        # pi_xyz_diff_concat: (B, 3+3+3+1, S, k) = (B, 10, S, k)
        pi_xyz_diff_concat = torch.cat((pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff), dim=1)
        
        # pi_feat_diff: (B, C1 + C2, S, k)
        pi_feat_diff = torch.cat((pi_points_expanded, qi_points_grouped), dim=1)
        # pi_feat1_new: (B, 10 + C1 + C2, S, k)
        pi_feat1_new = torch.cat((pi_xyz_diff_concat, pi_feat_diff), dim=1)

        # pi_feat1_new
        # the first flow embedding uses 3D Euclidean space information `pi_xyz_diff_concat` and
        # the features from the two frames of points clouds: `pi_points_expanded` and `qi_points_grouped`

        # first flow embeddings = MLP(pi_feat1_new): (B, mlp1[-1], S, k)
        pi_feat1_new = self.mlp_convs(pi_feat1_new)
        """
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            pi_feat1_new = F.relu(bn(conv(pi_feat1_new)))
        """


        # The spatial structure information `pi_xyz_diff_concat` not only helps
        # to determine the similarity of points, but also can contribute to deciding 
        # soft aggregation weights of the queried points

        # pi_xyz_encoding = FC(pi_xyz_diff_concat)
        pi_xyz_encoding = self.mlp_conv_xyz_1(pi_xyz_diff_concat)
        ##pi_xyz_encoding = F.relu(self.mlp_bn_xyz(self.mlp_conv_xyz(pi_xyz_diff_concat)))

        # pi_concat: (B, 2 * mlp1[-1], S, k)
        pi_concat = torch.cat((pi_xyz_encoding, pi_feat1_new), dim = 1)

        # WQ are the attentive weights learning: (B, mlp2[-1], S, k)
        # WQ = softmax(MLP(FC(pi_xyz_diff_concat), first flow embeddings))
        pi_concat = self.mlp2_convs(pi_concat)
        WQ = F.softmax(pi_concat, dim=3)
            
        # `pi_feat1_new` are The first attentive flow embeddings
        # (B, mlp[-1], S) where mlp[-1] == mlp1[-1] == mlp2[-1]
        pi_feat1_new = WQ * pi_feat1_new
        pi_feat1_new = torch.sum(pi_feat1_new, dim=3, keepdim=False)


        ### -----------------------------------------------------------
        ### SECOND AGGREGATE

        # `idx`: (B, S, m) 
        # we find the neighborhood of each 3d point in the first frame
        _, idx = pt_utils.knn_point(self.nsample, warped_xyz_t, warped_xyz_t)
        # pc_xyz_grouped: (B, 3, S, m)
        pc_xyz_grouped = pointutils.grouping_operation(warped_xyz, idx)
        # pc_points_grouped: (B, mlp[-1], S, m)
        pc_points_grouped = pointutils.grouping_operation(pi_feat1_new, idx)

        # pc_xyz_new: (B, 3, S, m)
        pc_xyz_new = torch.tile( torch.unsqueeze(warped_xyz, 3), [1, 1, 1, self.nsample] )
        # pc_points_new: (B, C1, S, m)
        pc_points_new = torch.tile( torch.unsqueeze(warped_points, 3), [1, 1, 1, self.nsample] )

        # pc_xyz_diff: (B, 3, S, m)
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new
        # pc_euc_diff: (B, 1, S, m)
        pc_euc_diff = torch.sqrt(torch.sum(torch.square(pc_xyz_diff), dim=1, keepdim=True) + 1e-20)
        # pc_xyz_diff_concat: (B, 10, S, m)
        # the 3D Euclidean space information
        pc_xyz_diff_concat = torch.cat((pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff), dim=1)

        # pc_xyz_encoding = FC(pc_xyz_diff_concat): (B, mlp1[-1], S, m)
        pc_xyz_encoding = self.mlp_conv_xyz_2(pc_xyz_diff_concat)
        ##pc_xyz_encoding = F.relu(self.mlp_bn_xyz(self.mlp_conv_xyz(pc_xyz_diff_concat)))

        # pc_concat: (B, mlp1[-1] + C1 + mlp[-1], S, m)
        pc_concat = torch.cat((pc_xyz_encoding, pc_points_new, pc_points_grouped), dim = 1)

        # WP = softmax(MLP(FC(pc_xyz_diff_concat), pc_points_new, pc_points_grouped))
        # WP are the second attentive wights: (B, mlp2[-1], S, m)
        pc_concat = self.mlp3_convs(pc_concat)
        WP = F.softmax(pc_concat, dim=3)

        # The final attentive flow embedding
        # pc_feat1_new: (B, mlp[-1], S)
        # mlp[-1] = mlp1[-1] = mlp2[-1]

        pc_feat1_new = WP * pc_points_grouped
        pc_feat1_new = torch.sum(pc_feat1_new, dim=3, keepdim=False)

        return pc_feat1_new