from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pytorch_utils as pt_utils


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetSAModulePWCLONet(nn.Module):
    def __init__(
        self, mlp: List[int], npoint: int, nsample: int, bn: bool = True
    ):
        # type: (PointnetSAModulePWCLONet, List[int], int, int, bool) -> None
        
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample

        mlp_spec = mlp
        if mlp[0] == 0:    # in case features is None
            mlp_spec[0] += 3
        
        mlp_spec[0] += 3        # because we always use xyz       
        
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn, init=torch.nn.init.xavier_uniform_)


    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()  # [B, 3, N]

        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)    # [B, npoint]
            )
            .transpose(1, 2)
            .contiguous()
        )   # [B, npoint, 3]

        # /!\ WE SHOULD SWITCH THE CONDITION !!!!!
        if features is not None:
            _, idx_q = pt_utils.knn_point(self.nsample, xyz, new_xyz)   # [B, npoint, nsample]

            grouped_xyz = pointnet2_utils.grouping_operation(xyz_flipped, idx_q)  # [B, 3, npoint, nsample]

            # IF features IS NONE, THAT WOULDN'T CREATE AN ERROR? 
            grouped_features = pointnet2_utils.grouping_operation(features, idx_q)  # [B, C, npoint, nsample]

            new_xyz_t = torch.permute(new_xyz, (0, 2, 1)).contiguous()      # [B, 3, npoint]
            new_xyz_expanded = torch.tile(new_xyz_t.unsqueeze(-1), (1, 1, 1, self.nsample)) # [B, 3, npoint, nsample]

            xyz_diff = grouped_xyz - new_xyz_expanded   # [B, 3, npoint, nsample]

            new_features = torch.cat((xyz_diff, grouped_features), dim=1)   # [B, C+3, npoint, nsample]

        else:
            _, idx_q = pt_utils.knn_point(self.nsample, xyz, new_xyz)   # [B, npoint, nsample]

            grouped_xyz = pointnet2_utils.grouping_operation(xyz_flipped, idx_q)  # [B, 3, npoint, nsample]

            new_xyz_t = torch.permute(new_xyz, (0, 2, 1)).contiguous()      # [B, 3, npoint]
            new_xyz_expanded = torch.tile(new_xyz_t.unsqueeze(-1), (1, 1, 1, self.nsample)) # [B, 3, npoint, nsample]
            xyz_diff = grouped_xyz - new_xyz_expanded   # [B, 3, npoint, nsample]

            new_features = torch.cat((xyz_diff, grouped_xyz), dim=1)    # (batch_size, 6, npoint, nample)

        #################################################################3
        
        new_features = self.mlp_module(new_features)  # (B, mlp[-1], npoint, nsample)

        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlp[-1], npoint, 1)

        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        return new_xyz, new_features



class PointnetFPModule(nn.Module):
    r"""Propagates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm

    Input
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of the xyz positions of the unknown features
    known : torch.Tensor
        (B, m, 3) tensor of the xyz positions of the known features
    unknow_feats : torch.Tensor
        (B, C1, n) tensor of the features to be propigated to
    known_feats : torch.Tensor
        (B, C2, m) tensor of features to be propigated

    Returns
    -------
    new_features : torch.Tensor
        (B, mlp[-1], n) tensor of the features of the unknown features
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)    # [B, n, 3], [B, n, 3]
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # [B, C1, n]
            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointnetLFPModuleMSG(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    learnable feature propagation layer.'''

    def __init__(
            self,
            *,
            mlps: List[List[int]],
            radii: List[float],
            nsamples: List[int],
            post_mlp: List[int],
            bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        super().__init__()

        assert(len(mlps) == len(nsamples) == len(radii))
        
        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                    # -- ME --, sample_uniformly=sample_uniformly)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor,
                features2: torch.Tensor, features1: torch.Tensor) -> torch.Tensor:
        r""" Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \sum_k(mlps[k][-1]), N2) tensor of the new_features descriptors
        """
        new_features_list = []

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz1, xyz2, features1
            )  # (B, C1, N2, nsample)
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], N2, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], N2, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], N2)

            if features2 is not None:
                new_features = torch.cat([new_features, features2],
                                           dim=1)  #(B, mlp[-1] + C2, N2)

            new_features = new_features.unsqueeze(-1)
            new_features = self.post_mlp(new_features)  # (B, post_mlp[-1], N2)

            new_features_list.append(new_features)

        return torch.cat(new_features_list, dim=1).squeeze(-1)
    

class PointnetFPModulePWCLONet(nn.Module):
    r""" 
    Propagate features from xyz1 to xyz2.

        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \sum_k(mlps[k][-1]), N1) tensor of the new_features descriptors
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            radius: float,
            nsample: int,
            post_mlp: List[int],
            bn: bool = True,
            use_xyz: bool = True,
            knn: bool = False,
            sample_uniformly: bool = False
    ):
        super().__init__()
        
        self.nsample = nsample
        self.knn = knn
        self.use_xyz = use_xyz

        mlp_spec = mlp
        if use_xyz:
            mlp_spec[0] += 3
        self.mlp = pt_utils.SharedMLP(mlp_spec, bn=bn, init=torch.nn.init.xavier_uniform_)
        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn, init=torch.nn.init.xavier_uniform_)

        self.grouper = pointnet2_utils.QueryAndGroup(radius, self.nsample, use_xyz=use_xyz)
                # -- ME --, sample_uniformly=sample_uniformly)
    

    def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor,
                features2: torch.Tensor, features1: torch.Tensor) -> torch.Tensor:
        r""" Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \sum_k(mlps[k][-1]), N2) tensor of the new_features descriptors
        """

        if self.knn:
            _, idx_q = pt_utils.knn_point(self.nsample, xyz1, xyz2)             # [B, N2, nsample]
            new_features = pointnet2_utils.grouping_operation(features1, idx_q) # [B, C1, N2, nsample]

            xyz1_t = torch.permute(xyz1, (0, 2, 1)).contiguous()            # [B, 3, N1]
            new_xyz_t = pointnet2_utils.grouping_operation(xyz1_t, idx_q)   # [B, 3, N2, nsample]

            xyz2_t = torch.permute(xyz2, (0, 2, 1)).contiguous()            # [B, 3, N2]
            xyz2_expanded = xyz2_t.unsqueeze(-1)                            # [B, 3, N2, 1]
            xyz_diff = new_xyz_t - xyz2_expanded                            # [B, 3, N2, nsample]

            if self.use_xyz:
                new_features = torch.cat((new_features, xyz_diff), dim=1)   # [B, C1+3, N2, nsample]

        else:
            new_features = self.grouper(
                xyz1, xyz2, features1
            )  # (B, C1, N2, nsample)

        new_features = self.mlp(
            new_features
        )  # (B, mlp[-1], N2, nsample)

        new_features = F.max_pool2d(
            new_features, kernel_size=[1, new_features.size(3)]
        )  # (B, mlp[-1], N2, 1)

        new_features = new_features.squeeze(-1)  # (B, mlp[-1], N2)

        if features2 is not None:
            new_features = torch.cat([new_features, features2],
                                        dim=1)  #(B, mlp[-1] + C2, N2)

        new_features = new_features.unsqueeze(-1)
        new_features = self.post_mlp(new_features)  # (B, post_mlp[-1], N2)

        return new_features.squeeze(-1)
