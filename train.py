from typing import Optional, Dict, Tuple, Union, List

from torch import nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR, ExponentialLR, CosineAnnealingLR
import torch
from torch.utils.data import Dataset, DataLoader

# Hydra and OmegaConf
import hydra
from hydra.conf import dataclass, MISSING
from omegaconf import OmegaConf

import argparse

import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib

import time
from tqdm import tqdm
import pprint
import numpy as np


project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from slam.common.pose                  import Pose
from slam.common.projection            import SphericalProjector
from slam.common.utils                 import assert_debug
from slam.dataset                      import DATASET, DatasetLoader, DatasetConfig
from slam.training.loss_modules        import _PointToPlaneLossModule, _PoseSupervisionLossModule, _PWCLONetLossModule, PWCLONetLossConfig
from slam.training.prediction_modules  import _PoseNetPredictionModule, _PWCLONetPredictionModule, PWCLONetPredictionConfig
from slam.training.trainer             import ATrainer, ATrainerConfig, TrainingConfig
from slam.common.torch_utils           import collate_fun
from slam.training.trainer             import AverageMeter

from slam.dataset.kitti_odometry_dataset import KittiOdometryConfig, KittiOdometryDataset
from slam.dataset.kitti_360_dataset_2  import Kitti360Dataset, KITTI360Config

from slam.common.kitti360_utils import KITTI360_TOOLS, KITTI360_IO, KITTI360_TRANSFORMATIONS
from evaluation                 import kittiOdomEval

from slam.models.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pytorch_utils import BNMomentumScheduler


try:
    import wandb
    USE_WANDB = True
except:
    print('Install Wandb for better logging')
    USE_WANDB = False


BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[best] learning_rate: {learning_rate:.4e}
[best] train_loss: {train_loss}
[best] train_rot_err: {train_rot_err}
[best] train_trans_err: {train_trans_err}
[best] eval_loss: {eval_loss}
[best] eval_rot_err: {eval_rot_err}
[best] eval_trans_err: {eval_trans_err}
[best] loss_s_param_trans: {loss_s_param_trans}
[best] loss_s_param_rot: {loss_s_param_rot}
"""



@dataclass
class PoseNetTrainingConfig(ATrainerConfig):
    """A Config for a PoseNetTrainer"""
    pose: str = "euler"
    ei_config: Optional[Dict] = None
    sequence_len: int = 2
    num_input_channels: int = 3

    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING

    network_name: str = 'posenet'

# ----------------------------------------------------------------------------------------------------------------------
# Trainer for PoseNet
class PoseNetTrainer(ATrainer):
    """Unsupervised / Supervised Trainer for the PoseNet prediction module"""

    def __init__(self, config: PoseNetTrainingConfig):
        super().__init__(config)
        self.pose = Pose(self.config.pose)

        if config.network_name == 'pwclonet':
            config.dataset.with_vertex_map = False
        
        self.dataset_config: DatasetLoader = DATASET.load(config.dataset)
        self.projector: SphericalProjector = self.dataset_config.projector()

        # Share root parameters to Prediction Node
        self.config.training.prediction.sequence_len = self.config.sequence_len
        self.config.training.prediction.num_input_channels = self.config.num_input_channels

    def __transform(self, data_dict: dict):
        return data_dict
    
    def load_scheduler(self) -> LRScheduler:
        return MultiStepLR(self._optimizer,
                        milestones=self.config.optimizer_scheduler_milestones,
                        gamma=self.config.optimizer_scheduler_decay,
                        last_epoch=self.num_epochs - 1)

    def prediction_module(self) -> nn.Module:
        """Returns the PoseNet Prediction Module"""
        return _PoseNetPredictionModule(OmegaConf.create(self.config.training.prediction), self.pose)

    def loss_module(self) -> nn.Module:
        """Return the loss module used to train the model"""
        loss_config = self.config.training.loss
        mode = loss_config.mode
        assert_debug(mode in ["unsupervised", "supervised"])
        if mode == "supervised":
            return _PoseSupervisionLossModule(loss_config, self.pose)
        else:
            return _PointToPlaneLossModule(loss_config, self.projector, self.pose)

    def load_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Loads the Datasets"""
        train_dataset, eval_dataset, test_dataset = self.dataset_config.get_sequence_dataset()
        train_dataset.sequence_transforms = self.__transform
        if test_dataset is not None:
            test_dataset.sequence_transforms = self.__transform
        if eval_dataset is not None:
            eval_dataset.sequence_transforms = self.__transform
        return train_dataset, eval_dataset, test_dataset

    def test(self):
        pass


class PWCLONetEexponentialScheduler(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, decay_clip=-1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.decay_clip = decay_clip

        if (self.decay_clip >= 1) or ((self.decay_clip < 0) and (self.decay_clip != -1)):
            raise RuntimeError('decay_clip should be either a float between ]0,1[ or -1')

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        return [max(group['lr'] * self.gamma, self.decay_clip) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.decay_clip) for base_lr in self.base_lrs]


@dataclass
class PWCLONetTrainingConfig(TrainingConfig):
    """A Config for training of a PoseNet module"""

    loss: PWCLONetLossConfig = MISSING
    prediction: PWCLONetPredictionConfig = MISSING


@dataclass
class PWCLONetConfig(ATrainerConfig):
    """A Config for a PWCLONetTrainer"""

    pose: str = "quaternions"
    sequence_len: int = 2
    num_input_channels: int = 3
    nb_levels: int = 4
    num_points: int = 8192
    device: str = 'cpu'

    dataset: DatasetConfig = MISSING
    training: PWCLONetTrainingConfig = MISSING

    optimizer_type: str = "adam"
    optimizer_learning_rate: float = MISSING # = 0.001
    optimizer_scheduler_decay = 0.7
    optimizer_beta = 0.9
    optimizer_momentum = 0.999

    scheduler_decay_clip: float = MISSING # = 0.00001
    coslr: bool = True

    bn_momentum_init: float = 0.5
    bn_decay_rate: float = 0.5
    bn_decay_step: int = 4 #20
    bn_momentum_max: float = 0.99

    network_name: str = 'pwclonet'


# ----------------------------------------------------------------------------------------------------------------------
# Trainer for PWCLONet
class PWCLONetTrainer(ATrainer):
    """Supervised Trainer for the PWCLONet prediction module"""

    def __init__(self, config: PWCLONetConfig):
        super().__init__(config)

        self.pose = Pose(self.config.pose)
        self.nb_levels = config.nb_levels
        self.num_points = config.num_points
        self.config = config

        self.config.dataset.num_points = self.num_points
        self.config.dataset.scalar_last = config.scalar_last
        self.config.dataset.velo_to_pose = config.velo_to_pose
        #self.dataset_config: DatasetLoader = DATASET.load(config.dataset)

        # Share root parameters to Prediction Node
        self.config.training.prediction.sequence_len        = self.config.sequence_len
        self.config.training.prediction.num_input_channels  = self.config.num_input_channels
        self.config.training.prediction.num_points          = self.config.num_points
        self.config.training.prediction.nb_levels           = self.config.nb_levels
        self.config.training.prediction.device              = self.config.device

        self.config.training.prediction.scalar_last         = self.config.dataset.scalar_last

        self.config.training.loss.nb_levels   = self.config.nb_levels
        self.config.training.loss.device      = self.config.device
        self.config.training.loss.scalar_last = self.config.dataset.scalar_last

        # self.config.scheduler_decay_clip = 0.000001  

        self.config.optimizer_type = "adam"
        # self.config.optimizer_learning_rate = 0.0001
        self.config.optimizer_scheduler_decay = 0.5
        self.config.optimizer_beta = 0.9
        self.config.optimizer_momentum = 0.999


    def init_reports(self):
        # templates
        self.best_report_template = BEST_REPORT_TEMPLATE

        self.best = {
            'best/epoch': 0,
            'best/learning_rate': self.config.optimizer_learning_rate,
            'best/train_loss': float("inf"),
            'best/train_rot_err': float("inf"),
            'best/train_trans_err': float("inf"),
            'best/eval_loss': float("inf"),
            'best/eval_rot_err': float("inf"),
            'best/eval_trans_err': float("inf"),
            'best/loss_s_param_trans': np.nan,
            'best/loss_s_param_rot': np.nan,
        }

        if USE_WANDB:
            wandb.login()

        self.empty_data()
        self.data_plot = {}
        self.fig_dir = os.path.join(self.log_dir, 'figs')
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)


    def load_experiment_config(self):
        return {
            "model_name": self.config.network_name,
            "epochs": self.nb_epochs,
            "dataset": self.config.dataset.dataset,
            "batch_size": self.config.batch_size,
            "train_sequences": self.config.dataset.train_sequences,
            "eval_sequences": self.config.dataset.eval_sequences,
            "test_sequences": self.config.dataset.test_sequences,
            "num_points": self.config.dataset.num_points,
            "optimizer_type": self.config.optimizer_type,
            "optimizer_learning_rate": self.config.optimizer_learning_rate,
            "optimizer_beta": self.config.optimizer_beta,
            "optimizer_momentum": self.config.optimizer_momentum,
            "loss_with_exp_weights": self.config.training.loss.with_exp_weights,
            "loss_init_weights": self.config.training.loss.init_weights,
            "loss_option": self.config.training.loss.loss_option,
            "log_dir": self.log_dir,
        }
    

    def load_scheduler(self) -> LRScheduler:
        if self.config.coslr:
            # T_max = min(self.config.num_epochs // 2, 20)
            T_max = self.config.num_epochs
            self.scheduler_name = f'CosineAnnealingLR[T_max:{T_max},eta_min:{self.config.scheduler_decay_clip:.6e}]'
            return CosineAnnealingLR(optimizer=self._optimizer, T_max=T_max, eta_min=self.config.scheduler_decay_clip)
        else:
            self.scheduler_name = f'ExponentialScheduler[gamma:{self.config.optimizer_scheduler_decay:.6e},decay_clip{self.config.scheduler_decay_clip:.6e}]'
            return PWCLONetEexponentialScheduler(optimizer=self._optimizer, gamma=self.config.optimizer_scheduler_decay, decay_clip=self.config.scheduler_decay_clip)
        

    def load_bn_scheduler(self) -> BNMomentumScheduler:
        self.bn_scheduler_name = f'BNMomentumScheduler-[init:{self.config.bn_momentum_init},rate:{self.config.bn_decay_rate},step:{self.config.bn_decay_step},max:{self.config.bn_momentum_max}]'
        bn_lbmd = lambda it: min(1 - self.config.bn_momentum_init * self.config.bn_decay_rate**(int(it / self.config.bn_decay_step)), self.config.bn_momentum_max)
        return BNMomentumScheduler(self.prediction_module_, bn_lambda=bn_lbmd, last_epoch=-1)


    def prediction_module(self) -> nn.Module:
        """Returns the PWCLONet Prediction Module"""
        return _PWCLONetPredictionModule(OmegaConf.create(self.config.training.prediction), self.pose)


    def loss_module(self) -> nn.Module:
        """Return the loss module used to train the model"""
        loss_config = self.config.training.loss
        return _PWCLONetLossModule(OmegaConf.create(loss_config), self.pose)


    def load_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Loads the Datasets"""
        if self.config.dataset.dataset == 'kitti_odometry':
            train_dataset, eval_dataset, test_dataset = KittiOdometryDataset(KittiOdometryConfig(**self.config.dataset), is_training=True), KittiOdometryDataset(KittiOdometryConfig(**self.config.dataset), is_training=False), KittiOdometryDataset(KittiOdometryConfig(**self.config.dataset), is_training=False, is_testing=True)
        elif self.config.dataset.dataset == 'kitti_360':
            train_dataset, eval_dataset, test_dataset = Kitti360Dataset(KITTI360Config(**self.config.dataset), is_training=True), Kitti360Dataset(KITTI360Config(**self.config.dataset), is_training=False), Kitti360Dataset(KITTI360Config(**self.config.dataset), is_training=False, is_testing=True)
        else:
            raise RuntimeError(f'[PWCLONetTrainer] Unknown dataset {self.config.dataset.dataset}')
        return train_dataset, eval_dataset, test_dataset
    

    def pred_loss_forward_pass(self, batch: Union[dict, List]):
        # Prediction step
        prediction_params, pred_dict = self.prediction_module_(batch)
        # Loss step
        gt_params = torch.cat((batch[3], batch[2]), 1)
        loss, log_dict = self.loss_module_(prediction_params, gt_params)

        for key in pred_dict.keys():
            if key not in log_dict.keys():
                log_dict[key] = pred_dict[key]

        return loss, log_dict, prediction_params


    def get_data(self, predicted_params, data_dict, mode: str = 'train'):
        """
        Store predictions for evaluation later
        """
        prediction_params_np = predicted_params.detach().cpu().numpy()

        assert prediction_params_np.shape[0] == data_dict[2].size(0), 'Sizes from prediction and batch are not matching'
        assert prediction_params_np.shape[0] == data_dict[3].size(0), 'Sizes from prediction and batch are not matching'

        assert prediction_params_np.shape[0] == data_dict[4].size(0), 'Sizes from prediction and batch are not matching'
        assert prediction_params_np.shape[0] == data_dict[5].size(0), 'Sizes from prediction and batch are not matching'

        batch_size = prediction_params_np.shape[0]

        gt_q = data_dict[2].detach().cpu().numpy().reshape(batch_size, 4)
        gt_t = data_dict[3].detach().cpu().numpy().reshape(batch_size, 3)

        seq = data_dict[4].detach().cpu().numpy().reshape(batch_size, 1).astype(int)
        frame = data_dict[5].detach().cpu().numpy().reshape(batch_size, 1).astype(int)

        pred_q = prediction_params_np[:, 0, 3:].reshape(batch_size, 4)
        pred_t = prediction_params_np[:, 0, :3].reshape(batch_size, 3)

        new_data = np.concatenate((seq, frame, gt_t, gt_q, pred_t, pred_q), axis=1)
        if len(self.data[mode]) > 0:
            self.data[mode] = np.concatenate((self.data[mode], new_data), axis=0)
        else:
            self.data[mode] = new_data


    # ----------------- EVALUATION -----------------

    def compute_cum_distances(self, trans):
        distances = np.linalg.norm(trans, axis=1)
        cum_distances = np.cumsum(distances)
        return cum_distances


    def compute_last_frame_idx(self, dist: Union[np.ndarray, List], first_frame_idx: int, len_: int):
        required_dist = dist[first_frame_idx] + len_

        new_dist = dist[first_frame_idx:]
        valid_frames = np.where(new_dist > required_dist)[0]

        last_frame_idx = -1
        if len(valid_frames) > 0:
            last_frame_idx = np.min(valid_frames) + first_frame_idx

        return last_frame_idx
    

    def compute_translation_error(self, gt_t, pred_t):
        total_gt_t = np.sum(gt_t, axis=0)
        total_pred_t = np.sum(pred_t, axis=0)

        trans_err = np.linalg.norm(total_gt_t - total_pred_t)
        return trans_err
    

    def compute_total_rot(self, rots):
        if len(rots.shape) == 3:
            total_rot = rots[0, :, :]
            for i in range(1, rots.shape[0]):
                total_rot = np.matmul(total_rot, rots[i, :, :])
        else:
            total_rot = rots

        return total_rot
    

    def rotationError(self, pose_error):
        """
        geodesic distance (deg)
        """
        EPS = 1e-7

        a = pose_error[0,0]
        b = pose_error[1,1]
        c = pose_error[2,2]
        d = 0.5*(a+b+c-1.0)

        return np.arccos(max(min(d, 1.0 - EPS), -1.0 + EPS)) * 180 / np.pi


    def compute_rotation_error(self, gt_q, pred_q):
        gt_rot = KITTI360_TOOLS.quat2mat_batch(gt_q, scalar_last=self.config.dataset.scalar_last)
        pred_rot = KITTI360_TOOLS.quat2mat_batch(pred_q, scalar_last=self.config.dataset.scalar_last)

        total_gt_rot = self.compute_total_rot(gt_rot)
        total_pred_rot = self.compute_total_rot(pred_rot)

        diff_rot = np.matmul(np.linalg.inv(total_pred_rot), total_gt_rot)
        return self.rotationError(diff_rot)


    def compute_metrics_epoch(self):
        modes = ['train', 'eval']
        lengths = [100, 200, 300, 400, 500, 600, 700, 800]

        metrics = {}
        avg_metrics = {}
        mode_counts = {}
        for mode in modes:
            metrics[mode] = {}
            avg_metrics[mode] = {}
            avg_metrics[mode]['translation'] = 0.
            avg_metrics[mode]['rotation'] = 0.
            mode_counts[mode] = 0
            sequences = np.unique(self.data[mode][:,0])
            for seq in sequences:
                metrics[mode][int(seq)] = {}
                seq_indexes = np.where(self.data[mode][:,0] == seq)[0]

                frames_indexes = np.argsort(self.data[mode][seq_indexes, 1])
                frames = self.data[mode][seq_indexes, 1][frames_indexes]

                cum_distances = self.compute_cum_distances(self.data[mode][seq_indexes,:][frames_indexes,2:5])

                seq_avg_trans_err = 0.
                seq_avg_rot_err = 0.
                nb_frames = 0
                for i in range(0, len(frames), 10):
                    frame_avg_trans_err = 0.
                    frame_avg_rot_err = 0.
                    nb_lengths = 0
                    for len_ in lengths:
                        last_frame_idx = self.compute_last_frame_idx(cum_distances, i, len_)
                        if last_frame_idx < 0:
                            continue

                        gt_t = self.data[mode][seq_indexes,:][frames_indexes,2:5][i:last_frame_idx, :]
                        pred_t = self.data[mode][seq_indexes,:][frames_indexes,9:12][i:last_frame_idx, :]
                        frame_avg_trans_err += self.compute_translation_error(gt_t, pred_t) / len_  # without unity

                        gt_t = self.data[mode][seq_indexes,:][frames_indexes,5:9][i:last_frame_idx, :]
                        pred_t = self.data[mode][seq_indexes,:][frames_indexes,12:16][i:last_frame_idx, :]
                        frame_avg_rot_err += self.compute_rotation_error(gt_t, pred_t) / len_       # deg / m

                        nb_lengths += 1

                    if nb_lengths > 0:
                        frame_avg_trans_err = frame_avg_trans_err / nb_lengths
                        frame_avg_rot_err = frame_avg_rot_err / nb_lengths

                        seq_avg_trans_err += frame_avg_trans_err
                        seq_avg_rot_err += frame_avg_rot_err

                        nb_frames += 1
                
                if nb_frames > 0:
                    seq_avg_trans_err = (seq_avg_trans_err / nb_frames) * 100     # %
                    seq_avg_rot_err = (seq_avg_rot_err / nb_frames) * 100         # deg / 100m

                    metrics[mode][int(seq)]['translation'] = seq_avg_trans_err
                    metrics[mode][int(seq)]['rotation'] = seq_avg_rot_err

                    avg_metrics[mode]['translation'] = avg_metrics[mode]['translation'] + seq_avg_trans_err
                    avg_metrics[mode]['rotation'] = avg_metrics[mode]['rotation'] + seq_avg_rot_err

                    mode_counts[mode] += 1
                else:
                    metrics[mode][int(seq)]['translation'] = -1.
                    metrics[mode][int(seq)]['rotation'] = -1.

        for mode in avg_metrics.keys():
            for metric in avg_metrics[mode].keys():
                if mode_counts[mode] > 0:
                    avg_metrics[mode][metric] = avg_metrics[mode][metric] / mode_counts[mode]
                else:
                    avg_metrics[mode][metric] = -1.

        return metrics, avg_metrics

    # ---------------------------------------------------

    def log_metrics(self):
        """
        Log the average errors to wandb and save figures of the translation and rotation errors
        """
        modes = ['train', 'eval']
        for mode in self.epoch_metrics.keys():
            for seq in self.epoch_metrics[mode].keys():
                for metric in self.epoch_metrics[mode][seq].keys():
                    if metric not in self.data_plot.keys():
                        self.data_plot[metric] = {}

                    label = f'{mode}_{seq}'
                    if label not in self.data_plot[metric].keys():
                        self.data_plot[metric][label] = {}
                        self.data_plot[metric][label]['x'] = []
                        self.data_plot[metric][label]['y'] = []
                    
                    self.data_plot[metric][label]['y'].append(self.epoch_metrics[mode][seq][metric])
                    self.data_plot[metric][label]['x'].append(int(self.num_epochs))

            for metric in self.avg_metrics[mode].keys():
                label = f'avg_{mode}'
                if label not in self.data_plot[metric].keys():
                    self.data_plot[metric][label] = {}
                    self.data_plot[metric][label]['x'] = []
                    self.data_plot[metric][label]['y'] = []
                
                self.data_plot[metric][label]['y'].append(self.avg_metrics[mode][metric])
                self.data_plot[metric][label]['x'].append(int(self.num_epochs))

        for metric in self.data_plot.keys():
            cmap_train = plt.get_cmap('Set2')
            cmap_eval = plt.get_cmap('Set3')
            colors_train = [cmap_train(i) for i in np.linspace(0, 1, len(self.data_plot[metric]))]
            colors_eval = [cmap_eval(i) for i in np.linspace(0, 1, len(self.data_plot[metric]))]
            fig = plt.figure(figsize=(8,8), dpi=110)
            for i, label in enumerate(self.data_plot[metric].keys()):
                marker = '.'
                if (label.split('_')[0] == 'train') or ('train' in label):
                    linestyle = '-'
                    color = colors_train[i]
                    linewidth = 1.
                else:
                    linestyle = ':'
                    color = colors_eval[i]
                    linewidth = 1.

                if ('avg' in label) or ('average' in label):
                    linewidth = 1.5
                    if (label.split('_')[0] == 'train') or ('train' in label):
                        linestyle = '-.'
                        
                    else:
                        linestyle = '--'

                if 'color' not in self.data_plot[metric][label].keys():
                    self.data_plot[metric][label]['color'] = color
                else:
                    color = self.data_plot[metric][label]['color']

                plt.plot(self.data_plot[metric][label]['x'], self.data_plot[metric][label]['y'], linestyle=linestyle, marker=marker, linewidth=linewidth, label=label, color=color) # linestyle=linestyle
                #plt.plot(plot_x, plot_y_t, 's-', color=cmap[i], label=method_name)
            if 'translation' in metric.lower():
                ylabel = "Translation Error %"
            else:
                ylabel = "Rotation Error deg/100m"
            plt.ylabel(ylabel)
            plt.xlabel("Epochs")
            plt.legend()
            plt.savefig(os.path.join(self.fig_dir, f'{metric}.png'))
            plt.close(fig=fig)


    def init_evaluators(self) -> dict:
        return {
            'loss_trans_l1': AverageMeter(),
            'loss_trans_l2': AverageMeter(),
            'loss_trans_l3': AverageMeter(),
            'loss_trans_l4': AverageMeter(),
            'loss_rot_l1': AverageMeter(),
            'loss_rot_l2': AverageMeter(),
            'loss_rot_l3': AverageMeter(),
            'loss_rot_l4': AverageMeter(),
            'loss_l1': AverageMeter(),
            'loss_l2': AverageMeter(),
            'loss_l3': AverageMeter(),
            'loss_l4': AverageMeter(),
            'log_counter': 0
        }


    def evaluate_metrics(self, batch, prediction_dict, log_dict, evaluators, batch_idx: int, end: bool = False, mode: str = 'train'):
        """
        Log average metrics from `init_evaluators`
        """
        if not end:
            for key in evaluators.keys():
                if key in log_dict.keys():
                    evaluators[key].update(log_dict[key].detach().cpu())

            if mode == 'train':
                log_freq = len(self.train_dataset) // 5
            else:
                log_freq = len(self.eval_dataset) // 5
            
            if USE_WANDB and evaluators['log_counter'] < 5: #(batch_idx % log_freq == 0):
                for key in log_dict.keys():
                    if 'embedding_mask' in key:
                        embedding_mask = log_dict[key].numpy()[0,:]
                        sorted_indices = np.argsort(embedding_mask)
                        cmap = matplotlib.colormaps['bwr']
                        #cmap = plt.cm.get_cmap('bwr', len(embedding_mask))
                        colors = np.array([list(cmap(i)[:3]) for i in range(len(embedding_mask))])
                        colors = colors[sorted_indices] * 255

                        for key in log_dict.keys():
                            if 'point_cloud' in key:
                                #point_cloud = batch[0][0,:self.num_points,:3]
                                #point_cloud = point_cloud.detach().cpu().numpy()

                                point_cloud = log_dict[key][0,:].numpy()

                                assert len(embedding_mask) == len(point_cloud), f'Embedding mask {embedding_mask.shape} and Point Cloud {point_cloud.shape} have different shapes'
                                
                                point_cloud = np.concatenate((point_cloud, colors), axis=1)
                                
                                self.wandb_run.log({
                                    f"visualization/{mode}/embedding_mask": wandb.Object3D(point_cloud),
                                    f"visualization/{mode}/embedding_mask_sequence": int(batch[4].detach().cpu().numpy()[0]),
                                    f"visualization/{mode}/embedding_mask_frame": int(batch[5].detach().cpu().numpy()[0]),
                                    f"visualization/{mode}/epoch": self.num_epochs,
                                })

                                evaluators['log_counter'] = evaluators['log_counter'] + 1
                                break

                        break
            
        else:
            for key in evaluators:
                if isinstance(evaluators[key], AverageMeter) and (evaluators[key].count > 0):
                    self.logging_dict[f'{mode}/{key}'] = evaluators[key].average
                else:
                    self.logging_dict[f'{mode}/{key}'] = -1.
    

    def load_extra_from_checkpoint(self, state_dict):
        if "scalar_last" in state_dict:
            self.config.scalar_last = state_dict["scalar_last"]
            self.config.dataset.scalar_last = state_dict["scalar_last"]
            self.config.training.loss.scalar_last = state_dict["scalar_last"]
        else:
            self.config.scalar_last = False
            self.config.dataset.scalar_last = False
            self.config.training.loss.scalar_last = False

        if "velo_to_pose" in state_dict:
            self.config.velo_to_pose = state_dict["velo_to_pose"]
            self.config.dataset.velo_to_pose = state_dict["velo_to_pose"]
        else:
            self.config.velo_to_pose = True
            self.config.dataset.velo_to_pose = True
    

    def save_extra_to_checkpoint(self, state_dict):
        state_dict["scalar_last"] = self.config.dataset.scalar_last
        state_dict["velo_to_pose"] = self.config.dataset.velo_to_pose

        return state_dict


    def load_dataset_info(self):
        self.experiment_config["scalar_last"] = self.config.dataset.scalar_last


    def load_scheduler_info(self):
        self.experiment_config["optimizer_scheduler_decay"] = self.config.optimizer_scheduler_decay
        self.experiment_config["optimizer_scheduler_name"] = self.scheduler_name


    def load_bn_info(self):
        self.experiment_config["bn_momentum_init"] = self.config.bn_momentum_init
        self.experiment_config["bn_decay_rate"] = self.config.bn_decay_rate
        self.experiment_config["bn_decay_step"] = self.config.bn_decay_step
        self.experiment_config["bn_momentum_max"] = self.config.bn_momentum_max
        self.experiment_config["bn_scheduler_name"] = self.bn_scheduler_name


    def _best_report(self):
        self.best['best/epoch'] = self.num_epochs
        self.best['best/learning_rate'] = self.config.optimizer_learning_rate
        self.best['best/train_loss'] = self.logging_dict['train/loss']
        self.best['best/train_rot_err'] = self.avg_metrics['train']['rotation']
        self.best['best/train_trans_err'] = self.avg_metrics['train']['translation']
        self.best['best/eval_loss'] = self.logging_dict['eval/loss']
        self.best['best/eval_rot_err'] = self.avg_metrics['eval']['rotation']
        self.best['best/eval_trans_err'] = self.avg_metrics['eval']['translation']

        if self.config.training.loss.with_exp_weights:
            self.best['best/loss_s_param_trans'] = self.loss_module_.exp_weighting.s_param.detach().cpu().numpy()[0]
            self.best['best/loss_s_param_rot'] = self.loss_module_.exp_weighting.s_param.detach().cpu().numpy()[1]
        
        best_report = self.best_report_template.format(
            epoch=self.best['best/epoch'],
            learning_rate=self.best['best/learning_rate'],
            train_loss=round(self.best['best/learning_rate'], 5),
            train_rot_err=round(self.best['best/train_rot_err'], 5),
            train_trans_err=round(self.best['best/train_trans_err'], 5),
            eval_loss=round(self.best['best/eval_loss'], 5),
            eval_rot_err=round(self.best['best/eval_rot_err'], 5),
            eval_trans_err=round(self.best['best/eval_trans_err'], 5),
            loss_s_param_trans=round(self.best['best/loss_s_param_trans'], 5),
            loss_s_param_rot=round(self.best['best/loss_s_param_rot'], 5)
        )

        self.wandb_run.log(self.best)
        self.log_string(best_report, mode='best')

        self.save_checkpoint(self.best_checkpoint_file)


    def test(self):
        pass


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


    def test_model(self):

        if self.test_dataset is None:
            raise RuntimeError('No Test Dataset')

        assert_debug(self.loss_module_ is not None)
        assert_debug(self.prediction_module_ is not None)

        self.eval_ = True
        self.prediction_module_.eval()
        self.loss_module_.eval()

        dataloader = DataLoader(
            self.test_dataset,
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fun)

        loss_meter = AverageMeter()
        progress_bar = self.progress_bar(dataloader, "Test")

        # ------- TIMERS -------
        epoch_start_time = time.time()
        data_start_time = time.time()
        data_loading_elapsed_time = 0.0
        batch_elapsed_time = 0.0

        data_tqdm = tqdm(total=0, position=1, bar_format='{desc}')
        batch_tqdm = tqdm(total=0, position=2, bar_format='{desc}')

        rel_pred_poses = {}
        rel_gt_poses = {}

        for batch_idx, batch in progress_bar:

            # send the data to the GPU
            batch = self.send_to_device(batch)

            # ------- TIMERS -------
            data_loading_elapsed_time += time.time() - data_start_time
            batch_start_time = time.time()

            # ------- LOGGING -------
            data_tqdm.set_description_str(f'[Test] Data Loading: {round(data_loading_elapsed_time/(batch_idx+1), 4)}')

            loss, log_dict, prediction_params = self.pred_loss_forward_pass(batch)

            # ------- TIMERS -------
            batch_elapsed_time += time.time() - batch_start_time

            # ------- LGGING -------
            batch_tqdm.set_description_str(f'[Eval] Batch {batch_idx} Processing: {round(batch_elapsed_time/(batch_idx+1), 4)}')

            # Log the log_dict
            self.log_dict(log_dict)
            if loss:
                loss_meter.update(loss.detach())

            self.eval_iter += 1

            # ------- TIMERS -------
            data_start_time = time.time()

            prediction_params_np = prediction_params.cpu().detach().numpy()

            assert prediction_params_np.shape[0] == len(batch[4]), 'Sizes from prediction and batch are not matching'
            assert prediction_params_np.shape[0] == len(batch[5]), 'Sizes from prediction and batch are not matching'

            for i in range(prediction_params_np.shape[0]):
                curr_seq = batch[4][i].cpu().detach().item()
                frame_idx = batch[5][i].cpu().detach().item()

                if curr_seq not in rel_pred_poses.keys():
                    rel_pred_poses[curr_seq] = {}
                    rel_gt_poses[curr_seq] = {}

                pred_params = prediction_params_np[i, 0, :]

                if self.config.dataset.dataset == 'kitti_odometry':
                    pred_q = pred_params[3:].reshape(4)
                    pred_t = pred_params[:3].reshape((3,1))
                    R = self.quat2mat(pred_q)

                    filler = np.array([0.0, 0.0, 0.0, 1.0])
                    filler = np.expand_dims(filler, axis = 0)   ##1*4
                    rel_pred_poses[curr_seq][frame_idx] = np.concatenate([np.concatenate([R, pred_t], axis=-1), filler], axis=0)
                    rel_pred_poses[curr_seq][frame_idx] = np.linalg.inv(rel_pred_poses[curr_seq][frame_idx])

                    # ----- GT -----
                    gt_q = batch[2][i,:].cpu().detach().numpy().reshape(4)
                    gt_t = batch[3][i,:].cpu().detach().numpy().reshape((3,1))

                    if self.config.dataset.scalar_last:
                        qq = np.copy(gt_q)
                        qq[0] = gt_q[-1]
                        qq[1:] = gt_q[:-1]
                        gt_q = np.copy(qq)

                    gt_R = self.quat2mat(gt_q)
                    rel_gt_poses[curr_seq][frame_idx] = np.concatenate([np.concatenate([gt_R, gt_t], axis=-1), filler], axis=0)
                    rel_gt_poses[curr_seq][frame_idx] = np.linalg.inv(rel_gt_poses[curr_seq][frame_idx])

                else:
                    pred_transformation = KITTI360_TOOLS.params2mat(pred_params, scalar_last=self.config.dataset.scalar_last)
                    rel_pred_poses[curr_seq][frame_idx] = pred_transformation

                    gt_q = batch[2][i,:].cpu().detach().numpy().reshape(-1)
                    gt_t = batch[3][i,:].cpu().detach().numpy().reshape(3)
                    gt_params = np.concatenate([gt_t, gt_q], axis=0)
                    gt_transformation = KITTI360_TOOLS.params2mat(gt_params, scalar_last=self.config.dataset.scalar_last)
                    rel_gt_poses[curr_seq][frame_idx] = gt_transformation

        
        pred_dir = 'predictions'
        job_name = 'pwclonet'
        pred_save_dir = os.path.join(pred_dir, job_name)
        gt_save_dir = 'gt'

        if not os.path.exists(pred_save_dir):
            os.makedirs(pred_save_dir)
        if not os.path.exists(gt_save_dir):
            os.makedirs(gt_save_dir)

        abs_pred_poses = {}
        abs_gt_poses = {}

        pwclonet_pred_dir = 'pwclonet_pred'
        pwclonet_gt_dir = 'pwclonet_gt'
        if not os.path.exists(pwclonet_pred_dir):
            os.makedirs(pwclonet_pred_dir)
        if not os.path.exists(pwclonet_gt_dir):
            os.makedirs(pwclonet_gt_dir)
        # -----------------------------------
        
        for seq in rel_pred_poses.keys():
            pred_seq_save_path = os.path.join(pred_save_dir, str(seq).zfill(2) + '_pred_rel.txt')
            gt_seq_save_path = os.path.join(gt_save_dir, str(seq).zfill(2) + '_gt_rel.txt')

            KITTI360_IO.save_poses(rel_pred_poses[seq], pred_seq_save_path)
            KITTI360_IO.save_poses(rel_gt_poses[seq], gt_seq_save_path)

            abs_pred_poses[seq] = {}
            abs_gt_poses[seq] = {}

            if self.config.dataset.dataset == 'kitti_odometry':
                velo_to_pose = False
            else:
                velo_to_pose = self.config.velo_to_pose

            abs_pred_poses[seq] = KITTI360_TRANSFORMATIONS.convert_to_absolute(rel_pred_poses[seq], velo_to_pose=False) #self.config.dataset.velo_to_pose)
            abs_gt_poses[seq] = KITTI360_TRANSFORMATIONS.convert_to_absolute(rel_gt_poses[seq], velo_to_pose=False) #self.config.dataset.velo_to_pose)

            pred_seq_save_path = os.path.join(pred_save_dir, str(seq).zfill(2) + '_pred.txt')
            gt_seq_save_path = os.path.join(gt_save_dir, str(seq).zfill(2) + '_gt.txt')

            KITTI360_IO.save_poses(abs_pred_poses[seq], pred_seq_save_path)
            KITTI360_IO.save_poses(abs_gt_poses[seq], gt_seq_save_path)

            # from PWCLONet ----------------------

            
            frames = sorted(list(abs_pred_poses[seq]))
            pwclonet_pred_pose = np.zeros((len(frames),12))
            pwclonet_gt_pose = np.zeros((len(frames),12))
            for i, frame in enumerate(frames):
                pwclonet_pred_pose[i,:] = abs_pred_poses[seq][frame][:3,:].reshape(12)
                pwclonet_gt_pose[i,:] = abs_gt_poses[seq][frame][:3,:].reshape(12)

            pwclonet_pred_seq_save_path = os.path.join(pwclonet_pred_dir, str(seq).zfill(2) + '_pred.txt')
            pwclonet_gt_seq_save_path = os.path.join(pwclonet_gt_dir, str(seq).zfill(2) + '.txt')

            np.savetxt(pwclonet_pred_seq_save_path, pwclonet_pred_pose, fmt='%.08f')
            np.savetxt(pwclonet_gt_seq_save_path, pwclonet_gt_pose, fmt='%.08f')

            config = {
                'gt_dir': pwclonet_gt_dir,
                'result_dir': pwclonet_pred_dir,
                'eva_seqs': f"{str(seq).zfill(2)}_pred",
                'toCameraCoord': False                    
            }
            args = OmegaConf.create(config)
            pose_eval = kittiOdomEval(args)
            pose_eval.eval(toCameraCoord=args.toCameraCoord)

        print('Predicted poses are saved in \'', pred_save_dir)
        print('Ground Truth poses are saved in \'', gt_save_dir)

        self.eval_ = False
        if loss_meter.count > 0:
            print(f"Test average loss : {loss_meter.average}")
            self._average_eval_loss = loss_meter.average


        elapsed_time = {
            'epoch': time.time() - epoch_start_time,
            'data': data_loading_elapsed_time / len(dataloader),
            'batch': batch_elapsed_time / len(dataloader)
        }

        pprint.pprint(elapsed_time)

        progress_bar.close()
        data_tqdm.close()
        batch_tqdm.close()


@hydra.main(config_name="train_posenet", config_path="config")
def run_posenet(cfg: PoseNetTrainingConfig):
    trainer = PoseNetTrainer(PoseNetTrainingConfig(**cfg))
    # Initialize the trainer (Optimizer, Cuda context, etc...)
    trainer.init()

    if trainer.config.do_train:
        trainer.train(trainer.config.num_epochs)
    if trainer.config.do_test:
        trainer.test()


@hydra.main(config_name="train_pwclonet", config_path="config")
def run_pwclonet(cfg: PWCLONetConfig):
    trainer = PWCLONetTrainer(PWCLONetConfig(**cfg))
    # Initialize the trainer (Optimizer, Cuda context, etc...)
    trainer.init()

    if trainer.config.do_train:
        trainer.train(trainer.config.num_epochs, experiment_title="PWCLONET")
    if trainer.config.do_test:
        #trainer.test()
        trainer.test_model()


if __name__ == "__main__":
    name = 'pwclonet'
    if name == 'pwclonet':
        run_pwclonet()
    elif name == 'posenet':
        run_posenet()
    else:
        raise RuntimeError('Unknown network name. Availbale options: [pwclonet, posenet]')

