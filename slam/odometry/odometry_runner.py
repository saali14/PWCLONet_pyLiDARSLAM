import dataclasses
import logging
from pathlib import Path
from typing import Optional
import time

import os
import torch

from abc import ABC
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

# Hydra and OmegaConf imports
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field

# Project Imports
from slam.common.pose import Pose
from slam.common.torch_utils import collate_fun
from slam.common.utils import check_tensor, assert_debug, get_git_hash
from slam.dataset import DatasetLoader, DATASET
from slam.eval.eval_odometry import OdometryResults, FrameState, TrajectoryPlotter
from slam.dataset.configuration import DatasetConfig

from slam.slam import SLAMConfig, SLAM

from slam.viz import _with_cv2

if _with_cv2:
    import cv2

from functools import partial


@dataclass
class SLAMRunnerConfig:
    """The configuration dataclass"""

    # --------------------------------
    # SLAMConfig
    slam: SLAMConfig = MISSING
    dataset: DatasetConfig = MISSING

    # ------------------
    # Default parameters
    max_num_frames: int = -1  # The maximum number of frames to run on
    log_dir: str = field(default_factory=os.getcwd)
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pose: str = "euler"

    fail_dir: str = field(default_factory=os.getcwd)  # By default the fail_dir is the same directory
    move_if_fail: bool = False

    # ----------------
    # Debug parameters
    viz_num_pointclouds: int = 200
    debug: bool = True
    save_results: bool = True


# -------------
# HYDRA Feature
# Automatically casts the config as a SLAMConfig object, and raises errors if it cannot do so
cs = ConfigStore.instance()
cs.store(name="slam_config", node=SLAMRunnerConfig)


class SLAMRunner(ABC):
    """
    A SLAMRunner runs a LiDAR SLAM algorithm on a set of pytorch datasets,
    And if the ground truth is present, it evaluates the performance of the algorithm and saved the results to disk
    """

    def __init__(self, config: SLAMRunnerConfig):
        super().__init__()

        self.config: SLAMRunnerConfig = config

        # Pytorch parameters extracted
        self.num_workers = self.config.num_workers
        self.batch_size = 1
        self.log_dir = self.config.log_dir
        self.device = torch.device(self.config.device)
        self.pin_memory = self.config.pin_memory if self.device != torch.device("cpu") else False

        self.pose = Pose(self.config.pose)
        self.viz_num_pointclouds = self.config.viz_num_pointclouds

        # Dataset config
        dataset_config: DatasetConfig = self.config.dataset
        self.dataset_loader: DatasetLoader = DATASET.load(dataset_config)

        self.slam_config: SLAMConfig = self.config.slam

    def save_config(self):
        """Saves the config to Disk"""
        with open(str(Path(self.log_dir) / "config.yaml"), "w") as config_file:
            # Add the git hash to improve tracking of modifications
            config_dict = self.config.__dict__

            git_hash = get_git_hash()
            if git_hash is not None:
                config_dict["git_hash"] = git_hash
            config_dict["_working_dir"] = os.getcwd()
            config_file.write(OmegaConf.to_yaml(config_dict))

    def handle_failure(self):
        """Handles Failure cases of the SLAM runner"""
        # In case of failure move the current working directory and its content to another directory
        if self.config.move_if_fail:
            try:
                fail_dir = Path(self.config.fail_dir)
                assert_debug(fail_dir.exists(),
                             f"[SLAM] -- The `failure` directory {str(fail_dir)} does not exist on disk")
                current_dir: Path = Path(os.getcwd())

                if fail_dir.absolute() == current_dir.absolute():
                    logging.warning(
                        "The `fail_dir` variable points to the current working directory. It will not be moved.")
                    return

                destination_dir = fail_dir
                if not destination_dir.exists():
                    destination_dir.mkdir()

                shutil.move(str(current_dir), str(destination_dir))
                assert_debug(not current_dir.exists(), "Could not move current working directory")
            except (Exception, AssertionError, KeyboardInterrupt):
                logging.warning("[PyLIDAR-SLAM] Could not move the directory")

    # -- ME -- added `sequences`
    def run_odometry(self, sequences:list=None, max_num_frames:int=-1):
        """Runs the LiDAR Odometry algorithm on the different datasets"""
        try:
            # Load the Datasets
            datasets: list = self.load_datasets(sequences=sequences)

        except (KeyboardInterrupt, Exception) as e:
            self.handle_failure()
            raise

        for sequence_name, dataset in datasets:
            # -- ME --
            if (sequence_name.isdigit()) and (not int(sequence_name) in sequences):
                continue
            # Build dataloader
            dataloader = DataLoader(dataset,
                                    collate_fn=collate_fun,
                                    pin_memory=self.pin_memory,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers)

            # Load/Init the SLAM
            slam = self.load_slam_algorithm()
            if self.config.save_results:
                self.save_config()
            slam.init()

            elapsed = 0.0
            relative_ground_truth = self.ground_truth(sequence_name)

            def catch_exception():
                _relative_poses = slam.get_relative_poses()
                if _relative_poses is not None and len(_relative_poses) > 0:
                    self.save_and_evaluate(sequence_name, _relative_poses, None)
                print("[ERRROR] running SLAM : the estimated trajectory was dumped")
                self.handle_failure()

            try:
                start = time.time()
                early_stop = self.process_next_sequence(dataloader=dataloader, sequence_name=sequence_name, slam=slam, 
                                      relative_ground_truth=relative_ground_truth, max_num_frames=max_num_frames)
                # Measure the time spent on the processing of the next frame
                elapsed_sec = time.time() - start
                elapsed += elapsed_sec
                
                """
                pred_outputfile = str(Path(self.log_dir) / sequence_name / f"trajectory_{sequence_name}.png")
                trajectoryPlotter = TrajectoryPlotter(output_file=pred_outputfile, labels=['prediction'])
                trajectoryPlotter.animation(self.process_sequence, len(dataloader), len(dataloader),
                                            dataloader, sequence_name, slam, relative_ground_truth)

                if relative_ground_truth is not None:
                    gt_outputfile = str(self.log_dir_path / f"trajectory_{sequence_name}_with_gt.png")
                    gt_trajectoryPlotter = TrajectoryPlotter(sequeoutputfile=gt_outputfile, labels=['prediction', 'gt'], with_gt=True)
                    gt_trajectoryPlotter.animation(data_gen=self.process_sequence, interval=len(dataloader), save_count=len(dataloader))
                """

            except KeyboardInterrupt:
                catch_exception()
                raise
            except (KeyboardInterrupt, Exception, RuntimeError, AssertionError) as e:
                catch_exception()
                raise e

            if self.config.save_results:
                # Dump trajectory constraints in case of loop closure
                slam.dump_all_constraints(str(Path(self.log_dir) / sequence_name))

            # Evaluate the SLAM if it has a ground truth
            relative_poses = slam.get_relative_poses()
            check_tensor(relative_poses, [-1, 4, 4])
            if relative_ground_truth is not None:
                # -- ME --
                if early_stop:
                    relative_ground_truth = relative_ground_truth[:max_num_frames+1,:,:]
                check_tensor(relative_ground_truth, [relative_poses.shape[0], 4, 4])

            del slam
            del dataloader
            del trajectoryPlotter

            if self.config.save_results:
                self.save_and_evaluate(sequence_name, relative_poses, relative_ground_truth, elapsed=elapsed)


    def process_next_sequence(self, dataloader: DatasetLoader, sequence_name: str, slam: SLAM, 
                         relative_ground_truth: Optional[np.ndarray], max_num_frames: int = -1):
        
        # -- ME -- to detect early stop
        early_stop = False

        odo_results = OdometryResults(str(Path(self.log_dir) / sequence_name))
        lastFrame = 0
        frame_state = FrameState.FIRST_FRAME

        translation_steps = np.linalg.norm(relative_ground_truth[:, :3, 3], axis=1)
        cum_distances = np.cumsum(translation_steps)

        # -- ME -- log the total distance traveled

        assert(len(cum_distances) == len(dataloader))
        curr_nb_evals = 1

        elapsed = 0.0
        
        for b_idx, data_dict in self._progress_bar(dataloader, desc=f"Sequence {sequence_name}"):
            start = time.time()
            data_dict = self._send_to_device(data_dict)

            # Process next frame
            slam.process_next_frame(data_dict)

            # Measure the time spent on the processing of the next frame
            elapsed_sec = time.time() - start
            elapsed += elapsed_sec

            # -- ME -- write metrics every x frames
            if self.config.save_results and ((b_idx == len(dataloader)-1) or (cum_distances[b_idx] >= curr_nb_evals * OdometryResults._FRAME_GAP)):
                relative_poses = slam.get_relative_poses()
                check_tensor(relative_poses, [-1, 4, 4])

                self.save_and_evaluate(sequence_name, relative_poses, relative_ground_truth, elapsed=elapsed, currFrame=b_idx, lastFrame=lastFrame, odo_results=odo_results, frame_state=frame_state)
                lastFrame = b_idx + 1
                frame_state = FrameState.SPECIAL_FRAME

                curr_nb_evals += 1

            if 0 < self.config.max_num_frames <= b_idx:
                break

            # -- ME -- to stop odometry prediction
            if 0 < max_num_frames <= b_idx:
                early_stop = True
                break

        return early_stop


    def save_and_evaluate(self, sequence_name: str,
                          trajectory: np.ndarray,
                          ground_truth: Optional[np.ndarray],
                          elapsed: Optional[float] = None,
                          currFrame: int = -1,
                          lastFrame: int = -1,
                          frame_state: FrameState = FrameState.NORMAL_FRAME,
                          odo_results: OdometryResults = None):
        """Saves metrics and trajectory in a folder on disk"""

        assert(currFrame >= lastFrame)

        if odo_results is None:
            odo_results = OdometryResults(str(Path(self.log_dir) / sequence_name))
        
        if (currFrame >= 0) and (lastFrame >= 0):
            odo_results.add_frames(sequence_name,
                                 currFrame,
                                 lastFrame,
                                 trajectory,
                                 ground_truth,
                                 elapsed,
                                 frame_state=frame_state)
            
        else:
            odo_results.add_sequence(sequence_name,
                                 trajectory,
                                 ground_truth,
                                 elapsed)
            odo_results.close()


    @staticmethod
    def _progress_bar(dataloader: DataLoader, desc: str = ""):
        return tqdm(enumerate(dataloader, 0),
                    desc=desc,
                    total=len(dataloader),
                    ncols=120, ascii=True)

    def _send_to_device(self, data_dict: dict):
        output_dict: dict = {}
        for key, item in data_dict.items():
            if isinstance(item, torch.Tensor):
                output_dict[key] = item.to(device=self.device)
            else:
                output_dict[key] = item
        return output_dict

    # -- ME -- added `sequences`
    def load_datasets(self, sequences:list=None) -> list:
        """
        Loads the Datasets for which the odometry is evaluated

        Returns
        -------
        A list of pairs (sequence_name :str, dataset_config :Dataset)
        Where :
            sequence_name is the name of a sequence which will be constructed
        """
        self.num_workers = min(self.dataset_loader.max_num_workers(), self.num_workers)
        # -- ME -- 
        sequences_dict = {'train':sequences, 'eval':[], 'test':[]}
        train_dataset, _, _, _ = self.dataset_loader.sequences(sequences=sequences_dict)
        assert_debug(train_dataset is not None)
        pairs = [(train_dataset[1][idx], train_dataset[0][idx]) for idx in range(len(train_dataset[0]))]
        return pairs

    def load_slam_algorithm(self) -> SLAM:
        """
        Returns the SLAM algorithm which will be run
        """
        slam = SLAM(self.config.slam,
                    projector=self.dataset_loader.projector(),
                    pose=self.pose,
                    device=self.device,
                    viz_num_pointclouds=self.viz_num_pointclouds)
        return slam

    def ground_truth(self, sequence_name: str) -> Optional[np.ndarray]:
        """
        Returns the ground truth associated with the sequence
        """
        return self.dataset_loader.get_ground_truth(sequence_name)
