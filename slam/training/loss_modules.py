from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

# Hydra and OmegaConf
from hydra.conf import dataclass, MISSING, field

# Project Imports
from hydra.core.config_store import ConfigStore

from pyquaternion import Quaternion
from omegaconf import OmegaConf


import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

from slam.common.geometry      import compute_normal_map, projection_map_to_points
from slam.common.optimization  import _LS_SCHEME, _WLSScheme, PointToPlaneCost
from slam.common.pose          import Pose
from slam.common.projection    import Projector
from slam.common.utils         import assert_debug, check_tensor


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LossConfig:
    """Abstract Loss Config for training PoseNet"""
    mode: str = MISSING


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PointToPlaneLossConfig(LossConfig):
    """Unsupervised Point-to-Plane Loss Config"""
    mode: str = "unsupervised"

    least_square_scheme: Optional[Dict[str, Any]] = field(default_factory=lambda: dict(scheme="geman_mcclure",
                                                                                       sigma=0.5))


# ----------------------------------------------------------------------------------------------------------------------
# Point-To-Plane Loss Module for unsupervised training of PoseNet
class _PointToPlaneLossModule(nn.Module):
    """
    Point-to-Plane Loss Module
    """

    def __init__(self, config: PointToPlaneLossConfig, projector: Projector, pose: Pose):
        nn.Module.__init__(self)
        self.pose = pose
        self.projector = projector
        self.config = config
        self._ls_scheme = _LS_SCHEME.get(**self.config.least_square_scheme)

    def point_to_plane_loss(self,
                            vm_target,
                            vm_reference,
                            nm_reference,
                            pose_tensor,
                            data_dict: dict):
        """
        Computes the Point-to-Plane loss between a target vertex map and a reference

        Parameters
        ----------
        vm_target: torch.Tensor
            The vertex map tensor
        vm_reference: torch.Tensor
            The vertex map tensor
        nm_reference: torch.Tensor
            The normal map tensor
        pose_tensor: torch.Tensor
            The relative pose parameters or transform matrix to apply on the target point cloud
        data_dict:
            The dictionary to add tensor for logging

        Returns
        -------
        The point-to-plane loss between the reference and the target

        """
        b, _, h, w = vm_target.shape
        # -- ME -- `target_pc`: (b, h * w, 3) -> so `C` should be 3 in vertex_map
        target_pc = vm_target.permute(0, 2, 3, 1).reshape(b, h * w, 3)

        mask_vm = (target_pc.norm(dim=2, keepdim=True) != 0.0)
        pc_transformed_target = self.pose.apply_transformation(target_pc, pose_tensor)

        # Mask out null points which will be transformed by the pose
        pc_transformed_target = pc_transformed_target * mask_vm
        vm_transformed = self.projector.build_projection_map(pc_transformed_target, height=h, width=w)

        # Transform to point clouds to compute the point-to-plane error
        pc_transformed = projection_map_to_points(vm_transformed)
        pc_reference = projection_map_to_points(vm_reference)
        normal_reference = projection_map_to_points(nm_reference)

        mask = ~(normal_reference.norm(dim=-1) == 0.0)
        mask *= ~(pc_reference.norm(dim=-1) == 0.0)
        mask *= ~(pc_transformed.norm(dim=-1) == 0.0)
        mask = mask.detach().to(torch.float32)

        residuals = mask * ((pc_reference - pc_transformed) * normal_reference).sum(dim=-1).abs()

        cost = self._ls_scheme.cost(residuals, target_points=pc_transformed, reference_points=pc_reference)
        loss_icp = ((cost * cost).sum(dim=1) / mask.sum(dim=1)).mean()

        return loss_icp

    def forward(self, data_dict: dict):
        vertex_map = data_dict["vertex_map"]
        if "normal_map" not in data_dict:
            check_tensor(vertex_map, [-1, 2, 3, -1, -1])
            b, seq, _, h, w = vertex_map.shape
            normal_map = compute_normal_map(vertex_map.view(b * seq, 3, h, w)).view(b, seq, 3, h, w)
            data_dict["normal_map"] = normal_map
        normal_map = data_dict["normal_map"]

        b, s, _, h, w = vertex_map.shape
        assert_debug(s == 2)

        tgt_vmap = vertex_map[:, 1]
        ref_vmap = vertex_map[:, 0]
        ref_nmap = normal_map[:, 0]

        tgt_to_ref = data_dict["pose_params"]
        if tgt_to_ref.size(-1) != 4:
            tgt_to_ref = self.pose.build_pose_matrix(tgt_to_ref)

        # Compute the 3D Point-to-Plane
        loss_icp = self.point_to_plane_loss(tgt_vmap, ref_vmap, ref_nmap, tgt_to_ref, data_dict).mean()
        loss = loss_icp

        return loss, data_dict


# ----------------------------------------------------------------------------------------------------------------------
# ExponentialWeights for Supervised Loss Module
class ExponentialWeights(nn.Module):
    """
    A Module which exponentially weights different losses during training

    It holds parameters weigh the different losses.
    The weights change during training, as they are concerned by the the gradient descent
    For n losses, the computed loss is :
    $$ loss = \\sum_{k=1}^n loss_i * e^{s_i} + s_i $$

    Parameters
    ----------
    num_losses : int
        The number of losses (and parameters)
    init_weights : list
        The initial weights for the parameters
    """

    def __init__(self, num_losses: int, init_weights: list):
        nn.Module.__init__(self)
        assert_debug(len(init_weights) == num_losses)

        self.s_param = torch.nn.Parameter(torch.tensor(init_weights), requires_grad=True)
        self.num_losses = num_losses

    def forward(self, list_losses: list) -> Tuple[torch.Tensor, list]:
        """
        Computes the exponential weighing of the losses in list_losses

        Parameters
        ----------
        list_losses : list
            The losses to weigh. Expects a list of self.num_losses torch.Tensor scalars

        Returns
        -------
        tuple (torch.Tensor, list)
            The weighted loss, and the list of parameters
        """
        assert_debug(len(list_losses) == self.num_losses)

        s_params = []
        loss = 0.0

        for i in range(self.num_losses):
            loss_item = list_losses[i]
            s_param = self.s_param[i]
            exp_part_loss = loss_item * torch.exp(-s_param) + s_param
            loss += exp_part_loss
            s_params.append(s_param.detach())

        return loss, s_params


# ----------------------------------------------------------------------------------------------------------------------
# Config for Supervised Loss Module
@dataclass
class SupervisedLossConfig(LossConfig):
    """Config for the supervised loss module of PoseNet"""
    mode: str = "supervised"

    loss_degrees: bool = True  # Whether to express the rotation loss in degrees (True) or radians (False)

    # The weights of rotation and translation losses
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Exponential Weighting Params
    # Parameters for adaptive scaling of rotation and translation losses during training
    with_exp_weights: bool = False
    init_weights: List[float] = field(default_factory=lambda: [-3.0, -3.0])

    # Loss option (l1, l2)
    loss_option: str = "l2"
    device: str = "cpu"


# ----------------------------------------------------------------------------------------------------------------------
# Supervised Loss Module
class _PoseSupervisionLossModule(nn.Module):
    """
    Supervised Loss Module
    """

    def __init__(self, config: SupervisedLossConfig, pose: Pose):
        super().__init__()
        self.config = config
        self.pose = pose
        self.euler_pose = Pose("euler")

        self.exp_weighting: Optional[ExponentialWeights] = None
        self.weights: Optional[list] = None
        self.degrees = self.config.loss_degrees
        if self.config.with_exp_weights:
            self.exp_weighting = ExponentialWeights(2, self.config.init_weights)
        else:
            self.weights = self.config.loss_weights
            assert_debug(len(self.weights) == 2)
        loss = self.config.loss_option

        assert_debug(loss in ["l1", "l2"])
        self.loss_config = loss

    def __l1(self, x, gt_x):
        return (x - gt_x).abs().sum(dim=1).mean()

    def __loss(self, x, gt_x):
        if self.loss_config == "l1":
            return self.__l1(x, gt_x)
        elif self.loss_config == "l2":
            return ((x - gt_x) * (x - gt_x)).sum(dim=1).mean()
        else:
            raise NotImplementedError("")

    def forward(self, data_dict: dict) -> Tuple[torch.Tensor, dict]:
        pose_params = data_dict["pose_params"]
        ground_truth = data_dict["ground_truth"]

        if self.degrees:
            euler_pose_params = self.euler_pose.from_pose_matrix(self.pose.build_pose_matrix(pose_params))
            gt_params = self.euler_pose.from_pose_matrix(ground_truth)

            pred_degrees = (180.0 / np.pi) * euler_pose_params[:, 3:]
            gt_degrees = (180.0 / np.pi) * gt_params[:, 3:]
            loss_rot = self.__loss(pred_degrees, gt_degrees)

            data_dict["loss_rot_l1"] = self.__l1(pred_degrees, gt_degrees).detach()
        else:
            gt_params = self.pose.from_pose_matrix(ground_truth)
            loss_rot = self.__loss(gt_params[:, 3:], pose_params[:, 3:])
            data_dict["loss_rot_l1"] = self.__l1(gt_params[:, 3:], pose_params[:, 3:]).detach()

        loss_trans = self.__loss(pose_params[:, :3], gt_params[:, :3])
        loss_trans_l1 = self.__l1(pose_params[:, :3], gt_params[:, :3])

        data_dict["loss_rot"] = loss_rot
        data_dict["loss_trans"] = loss_trans
        data_dict["loss_trans_l1"] = loss_trans_l1

        loss = 0.0
        if self.exp_weighting:
            loss, s_param = self.exp_weighting([loss_trans, loss_rot])
            data_dict["s_rot"] = s_param[1]
            data_dict["s_trans"] = s_param[0]
        else:
            loss = loss_trans * self.weights[0] + loss_rot * self.weights[1]

        data_dict["loss"] = loss

        if self.exp_weighting:
            data_dict['s_param_trans'] = self.exp_weighting.s_param[0].detach().cpu()
            data_dict['s_param_rot'] = self.exp_weighting.s_param[1].detach().cpu()

        return loss, data_dict


# ----------------------------------------------------------------------------------------------------------------------
# Config for Supervised Loss Module
@dataclass
class PWCLONetLossConfig(LossConfig):
    """Config for the loss module of PWCLONet"""
    mode: str = "supervised"

    loss_degrees: bool = False  # Whether to express the rotation loss in degrees (True) or radians (False)

    # The weights of rotation and translation losses
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Exponential Weighting Params
    # Parameters for adaptive scaling of rotation and translation losses during training
    with_exp_weights: bool = MISSING
    init_weights: List[float] = field(default_factory=lambda: [0.0, -2.5])

    # Loss option (l1, l2, l2_norm)
    loss_option: str = MISSING # "l2_norm"

    nb_levels: int = 4
    device: str = "cpu"

    scalar_last: bool = MISSING


# ----------------------------------------------------------------------------------------------------------------------
# Supervised Loss Module
class _PWCLONetLossModule(nn.Module):
    """
    Supervised Loss Module
    """

    def __init__(self, config: PWCLONetLossConfig, pose: Pose):
        super().__init__()

        self.config = config
        self.pose = pose
        self.euler_pose = Pose("euler")

        self.exp_weighting: Optional[ExponentialWeights] = None
        self.weights: Optional[list] = None
        self.degrees = self.config.loss_degrees
        if self.config.with_exp_weights:
            self.exp_weighting = ExponentialWeights(2, self.config.init_weights)
        else:
            self.weights = self.config.loss_weights
            assert_debug(len(self.weights) == 2)
        loss = self.config.loss_option

        assert_debug(loss in ["l1", "l2", "l2_norm"])
        self.loss_config = loss

        self.nb_levels = config.nb_levels


    def __l1(self, x, gt_x):
        return torch.mean(torch.sum(torch.abs(x - gt_x), dim=-1, keepdim=True)+1e-10)
        #return (x - gt_x).abs().sum(dim=1).mean()
    

    def __l2(self, x, gt_x):
        return torch.mean(torch.sum((x - gt_x)*(x - gt_x), dim=-1, keepdim=True)+1e-10)
        #return ((x - gt_x) * (x - gt_x)).sum(dim=1).mean()
    

    def __l2_norm(self, x, gt_x):
        l2_norm = torch.mean(torch.sqrt(torch.sum((x - gt_x)*(x - gt_x), dim=-1, keepdim=True)+1e-10))
        return l2_norm
        #return ((x - gt_x) * (x - gt_x)).sum(dim=1).sqrt().mean()


    def __loss(self, x, gt_x):
        if self.loss_config == "l1":
            return self.__l1(x, gt_x)
        elif self.loss_config == "l2":
            return self.__l2(x, gt_x)
        elif self.loss_config == "l2_norm":
            return self.__l2_norm(x, gt_x)
        else:
            raise NotImplementedError("")
        

    def __trans_loss(self, x, gt_x):
        trans_loss = torch.mean(torch.sqrt((x - gt_x)*(x - gt_x)+1e-10))
        return trans_loss
        #return ((x - gt_x) * (x - gt_x)).sqrt().mean()


    def __norm(self, x):
        x_norm = x / (torch.sqrt(torch.sum(x*x, dim=-1, keepdim=True)+1e-10) + 1e-10)
        return x_norm
        #return x / (((x * x).sum(dim=1)+1e-10).sqrt()+1e-10)


    """def geodesic_distance(self, q, gt_q):
        if self.config.scalar_last:
            if len(q.shape) == 2:
                new_q = np.zeros((q.shape[0], q.shape[1]))
                new_q[:,:-1] = q[:,1:]
                new_q[:,-1] = q[:,0]
            elif len(q.shape) == 1:
                new_q = np.zeros((q.shape[0]))
                new_q[:-1] = q[1:]
                new_q[-1] = q[0]
            else:
                raise RuntimeError(f'[quat2mat_batch] Unrecognized shape of quaternions: {q.shape}')
        quat = Quaternion(q)
        gt_quat = Quaternion(gt_q)
        return Quaternion.distance(quat, gt_quat)"""


    def get_rot_params(self, rot_params):
        if self.config.scalar_last:
            return_rot_params = torch.clone(rot_params)
            return_rot_params[:,:-1] = rot_params[:,1:]
            return_rot_params[:,-1] = rot_params[:,0]
        else:
            return_rot_params = rot_params

        return return_rot_params


    def forward(self, pred_params, gt_params) -> Tuple[torch.Tensor, dict]:
        pred_params_1 = pred_params[:, 0, :]
        pred_params_2 = pred_params[:, 1, :]
        pred_params_3 = pred_params[:, 2, :]
        pred_params_4 = pred_params[:, 3, :]

        assert pred_params_1.size(1) == 7 # self.pose.num_params()
        assert pred_params_2.size(1) == 7 # self.pose.num_params()
        assert pred_params_3.size(1) == 7 # self.pose.num_params()
        assert pred_params_4.size(1) == 7 # self.pose.num_params()
        assert gt_params.size(1) == 7     # self.pose.num_params()

        batch = gt_params.size(0)

        assert pred_params_1.size(0) == batch
        assert pred_params_2.size(0) == batch
        assert pred_params_3.size(0) == batch
        assert pred_params_4.size(0) == batch

        log_dict = {}

        # --------- Extracting rotation and translation parameters for each level ---------

        rot_gt_params = gt_params[:, 3:]
        trans_gt_params = gt_params[:, :3]

        # rot_params_1 = self.__norm(self.get_rot_params(pred_params_1[:, 3:]))
        rot_params_1 = self.__norm(pred_params_1[:, 3:])
        trans_params_1 = pred_params_1[:, :3]

        # rot_params_2 = self.__norm(self.get_rot_params(pred_params_2[:, 3:]))
        rot_params_2 = self.__norm(pred_params_2[:, 3:])
        trans_params_2 = pred_params_2[:, :3]

        # rot_params_3 = self.__norm(self.get_rot_params(pred_params_3[:, 3:]))
        rot_params_3 = self.__norm(pred_params_3[:, 3:])
        trans_params_3 = pred_params_3[:, :3]

        # rot_params_4 = self.__norm(self.get_rot_params(pred_params_4[:, 3:]))
        rot_params_4 = self.__norm(pred_params_4[:, 3:])
        trans_params_4 = pred_params_4[:, :3]


        # --------- Calculating rotation and translation losses for each level ---------

        loss_rot_lvl_1 = self.__l2_norm(rot_params_1, rot_gt_params)
        loss_trans_lvl_1 = self.__trans_loss(trans_params_1, trans_gt_params)

        loss_rot_lvl_2 = self.__l2_norm(rot_params_2, rot_gt_params)
        loss_trans_lvl_2 = self.__trans_loss(trans_params_2, trans_gt_params)

        loss_rot_lvl_3 = self.__l2_norm(rot_params_3, rot_gt_params)
        loss_trans_lvl_3 = self.__trans_loss(trans_params_3, trans_gt_params)

        loss_rot_lvl_4 = self.__l2_norm(rot_params_4, rot_gt_params)
        loss_trans_lvl_4 = self.__trans_loss(trans_params_4, trans_gt_params)


        # --------- Logging ---------

        log_dict["loss_rot_l1"] = loss_rot_lvl_1
        log_dict["loss_trans_l1"] = loss_trans_lvl_1

        log_dict["loss_rot_l2"] = loss_rot_lvl_2
        log_dict["loss_trans_l2"] = loss_trans_lvl_2

        log_dict["loss_rot_l3"] = loss_rot_lvl_3
        log_dict["loss_trans_l3"] = loss_trans_lvl_3

        log_dict["loss_rot_l4"] = loss_rot_lvl_4
        log_dict["loss_trans_l4"] = loss_trans_lvl_4


        # --------- Calculating the loss for each level ---------

        if self.config.with_exp_weights and self.exp_weighting:
            loss_lvl_1, s_param = self.exp_weighting([loss_trans_lvl_1, loss_rot_lvl_1])
            log_dict["s_rot_l1"] = s_param[1]
            log_dict["s_trans_l1"] = s_param[0]

            loss_lvl_2, s_param = self.exp_weighting([loss_trans_lvl_2, loss_rot_lvl_2])
            log_dict["s_rot_l2"] = s_param[1]
            log_dict["s_trans_l2"] = s_param[0]

            loss_lvl_3, s_param = self.exp_weighting([loss_trans_lvl_3, loss_rot_lvl_3])
            log_dict["s_rot_l3"] = s_param[1]
            log_dict["s_trans_l3"] = s_param[0]

            loss_lvl_4, s_param = self.exp_weighting([loss_trans_lvl_4, loss_rot_lvl_4])
            log_dict["s_rot_l4"] = s_param[1]
            log_dict["s_trans_l4"] = s_param[0]
        else:
            loss_lvl_1 = loss_trans_lvl_1 * self.weights[0] + loss_rot_lvl_1 * self.weights[1]
            loss_lvl_2 = loss_trans_lvl_2 * self.weights[0] + loss_rot_lvl_2 * self.weights[1]
            loss_lvl_3 = loss_trans_lvl_3 * self.weights[0] + loss_rot_lvl_3 * self.weights[1]
            loss_lvl_4 = loss_trans_lvl_4 * self.weights[0] + loss_rot_lvl_4 * self.weights[1]


        # --------- Logging (again) ---------

        log_dict[f"loss_l1"] = loss_lvl_1
        log_dict[f"loss_l2"] = loss_lvl_2
        log_dict[f"loss_l3"] = loss_lvl_3
        log_dict[f"loss_l4"] = loss_lvl_4


        # --------- Calculating the model's loss ---------

        loss = 1.6*loss_lvl_4 + 0.8*loss_lvl_3 + 0.4*loss_lvl_2 + 0.2*loss_lvl_1
        #loss = 1.6*loss_lvl_1 + 0.8*loss_lvl_2 + 0.4*loss_lvl_3 + 0.2*loss_lvl_4


        # --------- Logging (one more !) ---------

        log_dict[f"loss"] = loss

        if self.config.with_exp_weights and self.exp_weighting:
            log_dict['s_param_trans'] = self.exp_weighting.s_param[0].detach().cpu()
            log_dict['s_param_rot'] = self.exp_weighting.s_param[1].detach().cpu()

        return loss, log_dict

        """if self.degrees:
            lvl_pose_params = torch.cat((trans_params, rot_params), dim=-1)
            euler_pose_params = self.euler_pose.from_pose_matrix(self.pose.build_pose_matrix(lvl_pose_params))
            
            pred_degrees = (180.0 / np.pi) * euler_pose_params[:, 3:]
            gt_degrees = (180.0 / np.pi) * gt_params[:, 3:]
            loss_rot_lvl = self.__loss(pred_degrees, gt_degrees)

        else:
            loss_rot_lvl = self.__loss(rot_params, gt_params[:, 3:])"""

# ------------------------------------------------------------
# Hydra -- Add the config group for the different Loss options
cs = ConfigStore.instance()
cs.store(group="training/loss", name="supervised", node=SupervisedLossConfig)
cs.store(group="training/loss", name="unsupervised", node=PointToPlaneLossConfig)
cs.store(group="training/loss", name="pwclonet", node=PWCLONetLossConfig)

