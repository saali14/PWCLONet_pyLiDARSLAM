from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, Union
import os
import numpy as np
import math
import logging

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Hydra and OmegaConf
from hydra.conf import dataclass, field, MISSING, Any
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import time

from pyquaternion import Quaternion

try:
    import wandb
    USE_WANDB = True
except:
    print('Install Wandb for better logging')
    USE_WANDB = False

import traceback

import os
import sys

project_path = os.getenv('PYLIDAR_SLAM_PWCLONET_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `PYLIDAR_SLAM_PWCLONET_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from slam.common.torch_utils   import send_to_device, collate_fun
from slam.common.utils         import assert_debug, get_git_hash
from slam.viz.visualizer       import ImageVisualizer
from slam.viz.color_map        import tensor_to_image

from slam.training.loss_modules        import LossConfig
from slam.training.prediction_modules  import PredictionConfig


class AverageMeter(object):
    """
    An util object which progressively computes the mean over logged values
    """

    def __init__(self):
        self.average = 0.0
        self.count = 0

    def update(self, loss: float):
        """Adds a new item to the meter"""
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().item()
        self.average = (self.average * self.count + loss) / (self.count + 1)
        self.count += 1


# ----------------------------------------------------------------------------------------------------------------------
# Training Config

@dataclass
class TrainingConfig:
    """A Config for training of a PoseNet module"""
    loss: LossConfig = MISSING
    prediction: PredictionConfig = MISSING


# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ATrainerConfig:
    """defaults: List[Any] = field(default_factory=lambda:  [
        "_self_",
        "PWCLONetConfig"
    ])"""
    

    """The configuration dataclass for a Trainer"""
    train_dir: str = MISSING  # The train directory

    # Configuration for the current run
    do_train: bool = True
    do_test: bool = True
    num_epochs: int = 100  # Number of epochs for the current run

    # Standard training parameters
    device: str = "cpu"
    num_workers: int = 0
    shuffle: bool = MISSING
    batch_size: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 5

    # IO : for saving/loading state dicts
    out_checkpoint_file: str = "checkpoint.ckp"
    in_checkpoint_file: Optional[str] = "checkpoint.ckp"

    # Optimizer params
    optimizer_type: str = "adam"
    optimizer_momentum: float = 0.9
    optimizer_beta: float = 0.999
    optimizer_weight_decay: float = 0.001  # /!\ Important for PoseNet training stability
    optimizer_learning_rate: float = 0.0001
    optimizer_sgd_dampening: float = 0.0
    optimizer_sgd_nesterov: float = False
    # Scheduler params
    optimizer_scheduler_decay: float = 0.5
    optimizer_scheduler_milestones: List[int] = field(default_factory=lambda: [20 * i for i in range(10)])

    # Logging and visualization fields
    visualize: bool = False
    test_frequency: int = 20  # The number of epochs before launching testing
    visualize_frequency: int = 20
    scalar_log_frequency: int = 20
    tensor_log_frequency: int = 200
    average_meter_frequency: int = 50
    # Keys of tensors to log to tensorboard and visualize
    log_image_keys: List[str] = field(default_factory=list)  # The keys in the data_dict to add to tensorboard as images
    log_scalar_keys: List[str] = field(default_factory=list)  # The keys in the data_dict to add to tensorboard as scalar
    log_histo_keys: list = field(default_factory=list)
    viz_image_keys: list = field(default_factory=list)

    network_name: str = 'pwclonet'
    scalar_last: bool = True
    velo_to_pose: bool = False

    bn_momentum_init: float = 0.5
    bn_decay_rate: float = 0.5
    bn_decay_step: float = 20
    bn_momentum_max: float = 0.01


# ----------------------------------------------------------------------------------------------------------------------
class ATrainer(ABC):
    """
    An abstract Trainer class is the backbone for training deep learning modules

    Each ATrainer child classes defines a prediction Module and a loss Module, and how the data should be loaded

    # TODO Rewrite to allow training on multiple GPUs
    """

    def __init__(self,
                 config: ATrainerConfig):

        self.config = config

        # Training state variables
        self.num_epochs = 0
        self.train_iter: int = 0
        self.eval_iter: int = 0
        self.eval_: bool = False
        self.train_: bool = False
        self.test_: bool = False
        self.test_frequency = 10
        self.nb_epochs: int = 0


        # Loggers and Visualizer
        self.logger: Optional[SummaryWriter] = None
        self.do_visualize: bool = False
        self.image_visualizer: Optional[ImageVisualizer] = None
        self.average_meter: Optional[AverageMeter] = None

        # -- Averages of the last training epochs
        self._average_train_loss: Optional[float] = None
        self._average_eval_loss: Optional[float] = None

        # -- ME --
        self._average_train_metric: Optional[np.ndarray] = None
        self._average_eval_metric: Optional[np.ndarray] = None

        # Load the params from the registered params
        self.__load_params()

        # -- Optimizer variables
        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[MultiStepLR] = None
        self._bn_scheduler: Optional[object] = None

        # -- Parameters that should be defined by child classes
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.loss_module_: Optional[nn.Module] = None
        self.prediction_module_: Optional[nn.Module] = None

        self.init_reports()
        self.logger = self._init_logger()

        self.wandb_run = None

        self.logging_dict = {}
        self.scheduler_name = ''
        self.bn_scheduler_name = ''


    def __load_params(self):
        # ATrainer Params

        # -- Load the directories / file paths, etc...
        train_dir = self.config.train_dir
        train_dir_path = Path(train_dir)
        assert_debug(train_dir is not None, "The key 'train_dir' must be specified in the training params")
        if not train_dir_path.exists():
            assert_debug(Path(train_dir_path.parent).exists(),
                         f"Both the train directory {train_dir} and its parent do not exist")
            train_dir_path.mkdir()
        assert_debug(train_dir_path.is_dir(), f"The directory 'train_dir':{train_dir} does not exist")

        in_checkpoint_file = self.config.in_checkpoint_file
        out_checkpoint_file = self.config.out_checkpoint_file
        assert_debug(out_checkpoint_file is not None)
        if in_checkpoint_file is not None:
            #self.input_checkpoint_file = str(train_dir_path / in_checkpoint_file)
            # -- ME --
            self.input_checkpoint_file = in_checkpoint_file
        
        self.checkpoint_dir = str(train_dir_path / 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.output_checkpoint_file = os.path.join(self.checkpoint_dir, out_checkpoint_file)
        self.best_checkpoint_file = os.path.join(self.checkpoint_dir, 'best.ckp')

        self.log_dir = train_dir

        # -- Load the training params
        self.device = torch.device(self.config.device)
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.config.device
        if 'cuda' in self.config.device:
            torch.cuda.set_device(self.device)


    def init(self):
        """
        Initializes the ATrainer

        It will in order :
            - load the training, validation and test dataset_config
            - load the prediction module and loss module
            - send the modules to the chosen device
            - load the optimizer
            - reload the input_checkpoints (if it was specified)

        """
        # Loads the datasets:  # -- ME -- `DatasetOfSequences`
        start = time.time()

        self.train_dataset, self.eval_dataset, self.test_dataset = self.load_datasets()

        self.loss_module_ = self.loss_module()
        self.prediction_module_ = self.prediction_module()

        self.loss_module_ = self.loss_module_.to(self.device)
        self.prediction_module_ = self.prediction_module_.to(self.device)

        self._optimizer = self._load_optimizer()

        # Load checkpoint file
        if self.input_checkpoint_file:
            self.load_checkpoint()

        if self._optimizer:
            self._scheduler = self.load_scheduler()

        self._bn_scheduler = self.load_bn_scheduler()

        # -- Copy the configuration file in the train dir
        train_dir_path = Path(self.config.train_dir)
        with open(str(train_dir_path / "config.yaml"), "w") as config_file:
            # Add the git hash to improve tracking of modifications
            config_dict = self.config.__dict__

            git_hash = get_git_hash()
            if git_hash is not None:
                config_dict["git_hash"] = git_hash
            config_dict["_working_dir"] = os.getcwd()
            config_file.write(OmegaConf.to_yaml(config_dict))

        print('\n[Init Model] Elapsed time:', time.time() - start, '\n')

        self.experiment_config = self.load_experiment_config()
        
        self.load_dataset_info()
        self.load_scheduler_info()
        self.load_bn_info()


    @abstractmethod
    def init_reports(self):
        raise NotImplementedError

    
    def empty_data(self):
        self.data = {'train': [], 'eval': []}
            

    def _load_optimizer(self) -> Optimizer:
        loss_module_params = {"params": self.loss_module_.parameters(),
                              "lr": self.config.optimizer_learning_rate}
        prediction_module_params = {
            "params": self.prediction_module_.parameters(),
            "lr": self.config.optimizer_learning_rate
        }
        optimizer_type = self.config.optimizer_type
        assert_debug(optimizer_type in ["adam", "adamw", "rmsprop", "sgd"])
        if optimizer_type == "adam":
            return torch.optim.Adam([prediction_module_params, loss_module_params],
                                    betas=(self.config.optimizer_beta, self.config.optimizer_momentum),
                                    weight_decay=self.config.optimizer_weight_decay)
        elif optimizer_type == "adamw":
            return AdamW([prediction_module_params, loss_module_params],
                         betas=(self.config.optimizer_beta, self.config.optimizer_momentum),
                         weight_decay=self.config.optimizer_weight_decay)
        elif optimizer_type == "sgd":
            return torch.optim.SGD([prediction_module_params,
                                    loss_module_params],
                                   lr=self.config.optimizer_learning_rate,  # -- ME -- it was self.learning_rate
                                   momentum=self.config.optimizer_momentum,
                                   weight_decay=self.config.optimizer_weight_decay,
                                   dampening=self.config.optimizer_sgd_nesterov)
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop([prediction_module_params,
                                        loss_module_params],
                                       lr=self.config.optimizer_learning_rate,  # -- ME -- it was self.learning_rate
                                       weight_decay=self.config.optimizer_weight_decay,
                                       momentum=self.config.optimizer_momentum)
        else:
            raise NotImplementedError("")


    def _get_logger(self, name, log_file):
        log_format = "[%(name)s - %(process)d] %(asctime)s - %(levelname)s: %(message)s"

        handler = logging.FileHandler(log_file, mode='w') # logging.handlers.WatchedFileHandler(log_file, mode='w')
        formatter = logging.Formatter(log_format) # logging.BASIC_FORMAT
        handler.setFormatter(formatter)
        
        # Initialize the class variable with logger object
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]
        return logger
    

    def _init_logger(self):
        if self.log_dir is None:
            return None
        
        train_log_file = os.path.join(self.log_dir, f'log_train.log')
        self.training_logger = self._get_logger('', train_log_file)

        best_log_file = os.path.join(self.log_dir, f'best.log')
        self.best_logger = self._get_logger('best_logger', best_log_file)

        return SummaryWriter(log_dir=self.log_dir)


    def log_string(self, out_str, progress_bar: tqdm = None, mode='train'):
        if mode == 'train':
            self.training_logger.info(f'({self.num_epochs}/{self.nb_epochs}): ' + out_str)
        elif mode == 'best':
            self.best_logger.info(out_str)

        if progress_bar is not None:
            progress_bar.write(out_str)

        
    def _close_logger(self):
        return
        #self.training_logger.close()


    def _init_visualizer(self) -> Optional[ImageVisualizer]:
        return ImageVisualizer(self.config.viz_image_keys, update_frequency=self.config.visualize_frequency)      


    @abstractmethod
    def get_data(self, predicted_params, data_dict, mode: str = 'train'):
        pass


    @abstractmethod
    def compute_metrics_epoch(self):
        return dict(), dict()


    @abstractmethod
    def log_metrics(self):
        pass


    def train(self, num_epochs: int = 10, experiment_title: str = 'Model'):
        """
        Runs num_epochs of training

        Parameters
        ----------
        num_epochs : int
            The number of epochs to launch
        """
        progress_bar = tqdm(range(num_epochs), desc=f"Training {num_epochs} new epochs", ncols=max(num_epochs, 100), position=0, leave=True)
        self.nb_epochs = num_epochs

        elapsed_time_epoch = 0.0
        elapsed_time_data = 0.0
        elapsed_time_batch = 0.0

        #epoch_tqdm = tqdm(total=0, position=1, bar_format='{desc}', leave=False)
        self.experiment_config['epochs'] = num_epochs

        if USE_WANDB:
            try:
                job_name = os.environ['JOB_NAME']
            except:
                job_name = ''
            self.wandb_run = wandb.init(
                project=experiment_title,
                name=job_name,
                config=self.experiment_config
            )
            self.wandb_run.watch(self.prediction_module_, criterion=self.loss_module_, log='all')

        best_train_loss = float("inf")
        for epoch_id in range(num_epochs):
            self.num_epochs = epoch_id

            self.logging_dict = {}
            self.epoch_metrics = {}
            self.avg_metrics = {}
            try:
                elapsed_time_train = self.train_epoch()
                elapsed_time_eval = self.evaluate_epoch()

                self.epoch_metrics, self.avg_metrics = self.compute_metrics_epoch()
                self.empty_data()
                
                # ------- TIMERS -------
                elapsed_time_epoch += elapsed_time_train['epoch'] + elapsed_time_eval['epoch']
                elapsed_time_data += elapsed_time_train['data'] + elapsed_time_eval['data']
                elapsed_time_batch += elapsed_time_train['batch'] + elapsed_time_eval['epoch']

                if epoch_id % 10 == 0 and epoch_id != 0:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, f'{epoch_id}.ckp'))
                
                if self.logging_dict['train/loss'] <= best_train_loss:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, f'best_train.ckp'))
                    best_train_loss = self.logging_dict["train/loss"]
                    msg = f'New best train at epoch {epoch_id} with loss {best_train_loss}'
                    self.log_string(msg)
                    self.log_string(msg, mode='eval')
                
                if epoch_id > 0 and epoch_id % self.test_frequency == 0:
                    self.test()
                
                if self._scheduler:
                    self._scheduler.step()

                    lr = self._scheduler.get_last_lr()[0]
                    if lr != self.config.optimizer_learning_rate:
                        self.log_string(f"Last learning rate : {lr}")
                        self.config.optimizer_learning_rate = lr

                if self._bn_scheduler:
                    last_momentum = self._bn_scheduler.last_momentum
                    self._bn_scheduler.step(self.num_epochs)

                    if last_momentum != self._bn_scheduler.last_momentum:
                        self.log_string(f"Last momentum : {self._bn_scheduler.last_momentum}")

                if USE_WANDB:
                    wandb_log_dict = {
                        'epoch/epoch': int(epoch_id),
                        'epoch/learning_rate': lr,
                        'epoch/elapsed_time': elapsed_time_train['epoch'] + elapsed_time_eval['epoch'],
                        'epoch/train_loss': self.logging_dict['train/loss'],
                        'epoch/eval_loss': self.logging_dict['eval/loss'],
                    }

                    for key in self.logging_dict.keys():
                        wandb_log_dict[key] = self.logging_dict[key]

                    if hasattr(self.config.training.loss, 'with_exp_weights') and self.config.training.loss.with_exp_weights:
                        wandb_log_dict['epoch/loss_s_param_trans'] = self.loss_module_.exp_weighting.s_param.detach().cpu().numpy()[0]
                        wandb_log_dict['epoch/loss_s_param_rot'] = self.loss_module_.exp_weighting.s_param.detach().cpu().numpy()[1]

                    if self._bn_scheduler:
                        wandb_log_dict['epoch/bn_momentum'] = self._bn_scheduler.last_momentum

                    for mode in self.avg_metrics.keys():
                        for metric in self.avg_metrics[mode].keys():
                            wandb_log_dict[f'epoch/{mode}_{metric}'] = self.avg_metrics[mode][metric]
                    
                    self.wandb_run.log(wandb_log_dict)
                    self.log_metrics()

                if self.logging_dict['eval/loss'] <= self.best['best/eval_loss']:
                    self.log_string(f'Saving best model at epoch {self.num_epochs}')
                    self._best_report()

                progress_bar.update(1)
                self.log_string("---------------------------------------\n")
            
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                self.log_string(f'[ERROR: {exc_type} in {fname} line {exc_tb.tb_lineno}] {str(e)}', progress_bar)
                self.log_string(traceback.format_exc(), progress_bar) # traceback.print_exc()
                # finish training
                self._finish()
                #epoch_tqdm.close()
                progress_bar.close()
                self._close_logger()
                exit()

        self.log_string(f'Average Elapsed Time per Epoch: {round(elapsed_time_epoch/num_epochs, 4)}', progress_bar)

        self._finish()

        progress_bar.close()
        progress_bar.close()
        self._close_logger()


    @staticmethod
    def progress_bar(dataloader: DataLoader, desc: str = "", position: int = 0, leave: bool = True):
        return tqdm(enumerate(dataloader, 0),
                    desc=desc,
                    total=len(dataloader),
                    ncols=120, ascii=True, position=position, leave=leave)


    def train_epoch(self):
        """
        Launches the training for an epoch
        """
        assert_debug(self.loss_module_ is not None)
        assert_debug(self.prediction_module_ is not None)

        self.prediction_module_.train()
        self.loss_module_.train()
        self.train_ = True

        if self.config.num_workers == 0:
            dataloader = DataLoader(
                self.train_dataset,
                pin_memory=self.config.pin_memory,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                collate_fn=collate_fun,
                shuffle=self.config.shuffle)

        else:
            dataloader = DataLoader(
                self.train_dataset,
                pin_memory=self.config.pin_memory,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
                collate_fn=collate_fun,
                shuffle=self.config.shuffle)

        loss_meter = AverageMeter()
        evaluators = self.init_evaluators()
        #metric_meter = AverageMeter()
        progress_bar = self.progress_bar(dataloader, desc=f"Training epoch n°{self.num_epochs}", position=2, leave=False)

        # ------- TIMERS -------
        epoch_start_time = time.time()
        data_start_time = time.time()
        data_loading_elapsed_time = 0.0
        batch_elapsed_time = 0.0

        data_tqdm = tqdm(total=0, position=3, bar_format='{desc}', leave=False)
        batch_tqdm = tqdm(total=0, position=4, bar_format='{desc}', leave=False)

        log_pred_freq = max(int(len(dataloader) // self.config.average_meter_frequency), 1)
        for batch_idx, batch in progress_bar:

            # Reinitialize the optimizer
            self._optimizer.zero_grad()
            # send the data to the GPU
            batch = self.send_to_device(batch)

            # ------- TIMERS -------
            data_loading_elapsed_time += time.time() - data_start_time 
            batch_start_time = time.time()

            # ------- LOGGING -------
            data_tqdm.set_description_str(f'[Train] Data Loading: {round(data_loading_elapsed_time / (batch_idx+1),4)}') #, 4)}')

            # -- ME -- Prediction & Loss step
            loss, log_dict, prediction_dict = self.pred_loss_forward_pass(batch)
            # Prediction step
            #prediction_dict = self.prediction_module_(batch)
            # Loss step
            #loss, log_dict = self.loss_module_(prediction_dict)

            self.get_data(prediction_dict, batch, mode='train')

            if loss is not None:
                if torch.any(torch.isnan(loss)):
                    if torch.is_tensor(prediction_dict):
                        self.log_string('[ERROR] Predictions')
                        self.log_string(np.array2string(prediction_dict.detach().cpu().numpy(), precision=5, seperator=' '))

                    self.log_string('[ERROR] Loss')
                    self.log_string(np.array2string(loss.detach().cpu().numpy(), precision=5, seperator=' '))

                    self.log_string('[ERROR] Loss is NaN.')
                    raise RuntimeError("\n[ERROR] Loss is NaN.\n")
                # Optimizer step 
                try:
                    loss.backward()
                    self._optimizer.step()
                except RuntimeError as e:
                    print("\n[ERROR] NaN During back progation... Good luck.\n")
                    raise e

                
                if (batch_idx > 0) and (batch_idx % log_pred_freq == 0):
                    loss_meter.update(loss.detach().cpu())

                    self.evaluate_metrics(batch, prediction_dict, log_dict, evaluators, batch_idx=batch_idx, end=False, mode='train')

            # ------- TIMERS -------
            batch_elapsed_time += time.time() - batch_start_time

            # ------- LGGING -------
            batch_tqdm.set_description_str(f'[Train] Batch {batch_idx} Processing: {round(batch_elapsed_time/(batch_idx+1), 4)}')
            

            self.log_dict(log_dict)
            self.train_iter += 1

            # ------- TIMERS -------
            data_start_time = time.time()

        # Save module to checkpoint
        self.train_ = False
        if loss_meter.count > 0:
            self.log_string(f"Train average loss : {loss_meter.average}")
            self._average_train_loss = loss_meter.average

            self.logging_dict['train/loss'] = loss_meter.average
        else:
            self.logging_dict['train/loss'] = np.nan

        self.evaluate_metrics(None, prediction_dict, log_dict, evaluators, batch_idx=-1, end=True, mode='train')

        self.logging_dict['train/epoch'] = self.num_epochs

        elapsed_time = {
            'epoch': time.time() - epoch_start_time,
            'data': data_loading_elapsed_time / len(dataloader),
            'batch': batch_elapsed_time / len(dataloader)
        }

        progress_bar.close()
        data_tqdm.close()
        batch_tqdm.close()

        return elapsed_time
    

    def evaluate_epoch(self):
        """
        Runs the evaluation for an epoch
        """
        if self.eval_dataset is None:
            return

        assert_debug(self.loss_module_ is not None)
        assert_debug(self.prediction_module_ is not None)

        self.eval_ = True
        self.prediction_module_.eval()
        self.loss_module_.eval()

        dataloader = DataLoader(
            self.eval_dataset,
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fun)

        loss_meter = AverageMeter()
        evaluators = self.init_evaluators()
        #metric_meter = AverageMeter()
        progress_bar = progress_bar = self.progress_bar(dataloader, f"Eval epoch n°{self.num_epochs}", position=2, leave=False)

        # ------- TIMERS -------
        epoch_start_time = time.time()
        data_start_time = time.time()
        data_loading_elapsed_time = 0.0
        batch_elapsed_time = 0.0

        data_tqdm = tqdm(total=0, position=3, bar_format='{desc}', leave=False)
        batch_tqdm = tqdm(total=0, position=4, bar_format='{desc}', leave=False)

        log_pred_freq = max(int(len(dataloader) // self.config.average_meter_frequency), 1)
        for batch_idx, batch in progress_bar:

            # send the data to the GPU
            batch = self.send_to_device(batch)

            # ------- TIMERS -------
            data_loading_elapsed_time += time.time() - data_start_time
            batch_start_time = time.time()

            # ------- LOGGING -------
            data_tqdm.set_description_str(f'[Eval] Data Loading: {round(data_loading_elapsed_time/(batch_idx+1), 4)}')

            # Prediction & Loss step
            loss, log_dict, prediction_dict = self.pred_loss_forward_pass(batch)
            # Prediction step
            #prediction_dict = self.prediction_module_(batch)
            # Loss step
            #loss, log_dict = self.loss_module_(prediction_dict)

            self.get_data(prediction_dict, batch, mode='eval')

            # ------- TIMERS -------
            batch_elapsed_time += time.time() - batch_start_time

            # ------- LGGING -------
            batch_tqdm.set_description_str(f'[Eval] Batch {batch_idx} Processing: {round(batch_elapsed_time/(batch_idx+1), 4)}')

            # Log the log_dict
            self.log_dict(log_dict)
            if loss:
                if (batch_idx > 0) and (batch_idx % log_pred_freq == 0):
                    loss_meter.update(loss.detach().cpu())

                    self.evaluate_metrics(batch, prediction_dict, log_dict, evaluators, batch_idx=batch_idx, end=False, mode='eval')

            self.eval_iter += 1

            # ------- TIMERS -------
            data_start_time = time.time()
            
        if loss_meter.count > 0:
            self.log_string(f"Eval average loss : {loss_meter.average}")
            self._average_eval_loss = loss_meter.average

            self.logging_dict['eval/loss'] = loss_meter.average
        else:
            self.logging_dict['eval/loss'] = np.nan

        self.evaluate_metrics(None, prediction_dict, log_dict, evaluators, batch_idx=-1, end=True, mode='eval')

        self.logging_dict['eval/epoch'] = self.num_epochs
        
        self.eval_ = False

        elapsed_time = {
            'epoch': time.time() - epoch_start_time,
            'data': data_loading_elapsed_time / len(dataloader),
            'batch': batch_elapsed_time / len(dataloader)
        }

        progress_bar.close()
        data_tqdm.close()
        batch_tqdm.close()

        return elapsed_time


    def pred_loss_forward_pass(self, batch: Union[dict, List]):
        # Prediction step
        if self.eval_:
            with torch.no_grad():
                prediction_dict = self.prediction_module_(batch)
        elif self.train_:
            prediction_dict = self.prediction_module_(batch)
        # Loss step
        loss, log_dict = self.loss_module_(prediction_dict)
    
        return loss, log_dict, prediction_dict


    @abstractmethod
    def init_evaluators(self):
        pass


    @abstractmethod
    def evaluate_metrics(self, batch, prediction_dict, log_dict, evaluators, batch_idx: int, end: bool = False, mode: str = 'train'):
        pass


    @abstractmethod
    def load_extra_from_checkpoint(self, state_dict) -> None:
        pass


    @abstractmethod
    def save_extra_to_checkpoint(self, state_dict) -> dict:
        return state_dict

    @abstractmethod
    def load_experiment_config(self) -> dict:
        return dict()


    @abstractmethod
    def load_dataset_info(self):
        pass


    @abstractmethod
    def load_scheduler_info(self):
        pass


    @abstractmethod
    def load_bn_info(self):
        pass
    

    def load_checkpoint(self, fail_if_absent: bool = False):
        """
        Loads a checkpoint file saved during training

        The checkpoint file is a python dictionary saved,
        The dictionary contains the parameters of the optimizer and
        The submodules of the loss module and the .train module

        To load the models, the variable input_checkpoint_file must be defined,
        and pointing to an existing checkpoint file

        Parameters
        ----------
        fail_if_absent : bool
            Whether to fail if the checkpoint file does not exist
        """

        if not self.input_checkpoint_file:
            return
        checkpoint_path = Path(self.input_checkpoint_file)
        if fail_if_absent:
            assert_debug(checkpoint_path.exists() and checkpoint_path.is_file(), "The checkpoint file does not exist")
        else:
            if not checkpoint_path.exists():
                return
        state_dict = torch.load(str(checkpoint_path), map_location=self.device)

        # Load the optimizer from the state dict
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self.loss_module_.load_state_dict(state_dict["loss_module"])
        self.prediction_module_.load_state_dict(state_dict["prediction_module"])
        self.num_epochs = state_dict["num_train_epochs"]
        self.eval_iter = state_dict["eval_iter"]
        self.train_iter = state_dict["train_iter"]
        if "average_eval_loss" in state_dict:
            self._average_eval_loss = state_dict["average_eval_loss"]
        if "average_train_loss" in state_dict:
            self._average_train_loss = state_dict["average_train_loss"]
        if "last_lr" in state_dict:
            self.config.optimizer_learning_rate = state_dict["last_lr"]
        if "best" in state_dict:
            self.best = state_dict["best"]

        self.load_extra_from_checkpoint(state_dict)

        print('\nModel succefully loaded from checkpoint file: \'' + os.path.basename(os.path.normpath(str(checkpoint_path))) + '\'\n', flush=True)


    def save_checkpoint(self, checkpoint_file):
        """
        Saves the modules and optimizer parameters in a checkpoint file
        """
        if not checkpoint_file:
            return

        state_dict = {
            "optimizer": self._optimizer.state_dict(),
            "loss_module": self.loss_module_.state_dict(),
            "prediction_module": self.prediction_module_.state_dict(),
            "num_train_epochs": self.num_epochs,
            "train_iter": self.train_iter,
            "eval_iter": self.eval_iter,
            "best": self.best,
        }
        if self._average_eval_loss is not None:
            state_dict["average_eval_loss"] = self._average_eval_loss
        if self._average_train_loss is not None:
            state_dict["average_train_loss"] = self._average_train_loss
        if self._scheduler:
            state_dict["last_lr"] = self._scheduler.get_last_lr()

        state_dict = self.save_extra_to_checkpoint(state_dict)

        torch.save(state_dict, checkpoint_file)


    def send_to_device(self, data_dict: dict) -> dict:
        """
        Default method to send a dictionary to a device

        By default, only tensors are sent to the GPU

        Parameters
        ----------
        data_dict : dict
            A dictionary of objects to send to the device
        """
        return send_to_device(data_dict, self.device, torchviz_conversion=False)


    def log_dict(self, log_dict: dict):
        """
        Logs the output of the

        Parameters
        ----------
        log_dict : dict
            The dictionary of items to be logged
        """
        if self.logger is None:
            # Init the logger
            self.logger = self._init_logger()

        # Init the visualizer
        if self.image_visualizer is None and self.do_visualize:
            self.image_visualizer = self._init_visualizer()
            if self.image_visualizer is None:
                self.do_visualize = False

        if self.logger is None:
            return

        assert_debug(self.train_ or self.eval_)
        if self.train_:
            _iter = self.train_iter
            tag_prefix = ".train/"
        else:
            _iter = self.eval_iter
            tag_prefix = ".eval/"

        if _iter % self.config.scalar_log_frequency == 0:
            # Log scalars
            for scalar_key in self.config.log_scalar_keys:
                assert_debug(scalar_key in log_dict, f"scalar key {scalar_key} not in log_dict")
                item = log_dict[scalar_key]
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self.logger.add_scalar(f"{tag_prefix}{scalar_key}", item, _iter)

        if _iter % self.config.tensor_log_frequency == 0:
            # Log histograms
            for histo_key in self.config.log_histo_keys:
                assert_debug(histo_key in log_dict)
                self.logger.add_histogram(f"{tag_prefix}{histo_key}", log_dict[histo_key], _iter)

            # Log images
            for image_key in self.config.log_image_keys:
                assert_debug(image_key in log_dict)
                image = tensor_to_image(log_dict[image_key])
                self.logger.add_image(f"{tag_prefix}{image_key}", image, _iter)

        if self.image_visualizer is not None:
            self.image_visualizer.visualize(log_dict, _iter)


    @abstractmethod
    def load_scheduler(self) -> LRScheduler:
        """
        Returns the scheduler for the specific trainer

        Returns
        -------
        LRScheduler

        """
        raise NotImplementedError


    @abstractmethod
    def load_bn_scheduler(self) -> LRScheduler:
        """
        Returns the scheduler for the specific trainer

        Returns
        -------
        LRScheduler

        """
        raise NotImplementedError


    @abstractmethod
    def prediction_module(self) -> nn.Module:
        """
        Returns the prediction module for the specific trainer

        Returns
        -------
        nn.Module
            The evaluation module which extracts data for the evaluation
            The module takes the dict with the data for each iter,
            And returns a dict with all the predictions and data expected by the loss Module or the evaluation

        """
        raise NotImplementedError()


    @abstractmethod
    def loss_module(self) -> nn.Module:
        """
        Returns the loss module for the specific trainer

        Returns
        -------
        nn.Module
            The loss module computes the loss on which will be applied the gradient descent
            The module should expect the dict returned from the prediction module
            And returns a tuple, with the first item the loss (as a torch.Tensor)
            And the second item a dictionary with all the data to be logged
        """

        raise NotImplementedError()


    @abstractmethod
    def load_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """
        Returns the .train, validation and test datasets as options
        """
        raise NotImplementedError()


    @abstractmethod
    def test(self):
        raise NotImplementedError()
    

    def _finish(self):
        """
            + Report best in `best.txt`
            + Save Checkpoint in `checkpoint.tar`
            + Save Last Model in `model_last.pth`
            + Save both train ad eval logs in `tensorboard/{train|eval}/all_scalars.json`
        """
        self.log_string("Training Completed...\n")

        # save check point
        self.log_string("saving checkpoint...\n")
        self.save_checkpoint(os.path.join(self.log_dir, 'last_checkpoint.ckp'))

        # save prediction model
        self.log_string("saving last prediction model...\n")
        last_pred_model_path = os.path.join(self.log_dir, "prediction_model_last.pth")
        torch.save(self.prediction_module_.state_dict(), last_pred_model_path)

        # save loss model
        self.log_string("saving last loss model...\n")
        last_loss_model_path = os.path.join(self.log_dir, "loss_model_last.pth")
        torch.save(self.loss_module_.state_dict(), last_loss_model_path)

        # export
        #self.logger.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))

        if USE_WANDB:

            try:
                job_name = os.environ['JOB_NAME']
            except:
                job_name = ''

            try:
                meta_data = self.experiment_config
                meta_data['final_learning_rate'] = self.config.optimizer_learning_rate

                final_pred_model = wandb.Artifact(
                    f"pred_model_{job_name}/{self.wandb_run.id}", 
                    type='model',
                    metadata=meta_data
                )
                final_pred_model.add_file(last_pred_model_path)
                self.wandb_run.save(last_pred_model_path)
                self.wandb_run.log_artifact(final_pred_model)

                final_loss_model = wandb.Artifact(
                    f"loss_model_{job_name}/{self.wandb_run.id}", 
                    type='model',
                    metadata=meta_data
                )
                final_loss_model.add_file(last_loss_model_path)
                self.wandb_run.save(last_loss_model_path)
                self.wandb_run.log_artifact(final_loss_model)
            except:
                pass
            
            self.wandb_run.finish()

        self.clean()


    def clean(self):
        pass


    @abstractmethod
    def _best_report(self):
        """
            Log best report
        """
        pass

