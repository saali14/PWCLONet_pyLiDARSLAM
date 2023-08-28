from abc import abstractmethod, ABC
from typing import Tuple, Optional

from torch.utils.data import Dataset, ConcatDataset

# Hydra and OmegaConf
from hydra.conf import dataclass
from omegaconf import MISSING

import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

# Project Imports
from pyLiDAR_SLAM.slam.common.projection        import SphericalProjector
from pyLiDAR_SLAM.slam.dataset.sequence_dataset import DatasetOfSequences


@dataclass
class DatasetConfig:
    """A DatasetConfig contains the configuration values used to define a DatasetConfiguration"""
    dataset: str = MISSING

    # The length of the sequence returned
    sequence_len: int = 2

    # ----------------------------------
    # Default item keys in the data_dict
    vertex_map_key: str = "vertex_map"
    numpy_pc_key: str = "numpy_pc"
    absolute_gt_key: str = "absolute_pose_gt"
    relative_gt_key: str = "relative_pose_gt"
    with_numpy_pc: bool = True  # Whether to add the numpy pc to the data_dict
    num_points: int = -1
    with_vertex_map: bool = True


class DatasetLoader(ABC):
    """
    A DatasetConfiguration is the configuration for the construction of pytorch Datasets
    """

    @classmethod
    def max_num_workers(cls):
        """Returns the maximum number of workers allowed by this dataset

        Note: Not respecting this constraint can lead to undefined behaviour for
              Datasets which do not support Random Access
        """
        return 20
    
    @staticmethod
    def relative_gt_key():
        """The key (in data_dict) for the relative_pose_gt"""
        return "relative_pose_gt"

    @staticmethod
    def absolute_gt_key():
        """The key (in data_dict) for the absolute_pose_gt"""
        return "absolute_pose_gt"

    @staticmethod
    def numpy_pc_key():
        """The key (in data_dict) for xyz pointcloud"""
        return "numpy_pc"
    
    @staticmethod
    def vertex_map_key():
        """The key (in data_dict) for vertex map"""
        return "vertex_map"

    def __init__(self, config: DatasetConfig):
        self.config = config


    @abstractmethod
    def projector(self) -> SphericalProjector:
        """
        Returns the Default Spherical Image projector associated to the dataset_config
        """
        raise NotImplementedError("")

    @abstractmethod
    def sequences(self):
        """
        Returns the train, eval and test datasets and the corresponding sequence name

        Returns: (train, eval, test, transform)
            train (Optional[List[Dataset], List]): Is an Optional pair of a list of datasets
                                                   and the corresponding sequences
            eval (Optional[List[Dataset], List]): Idem
            test (Optional[List[Dataset], List]): Idem
            transform (callable): The function applied on the data from the given datasets

        """
        raise NotImplementedError("")

    def get_dataset(self) -> Tuple[Dataset, Dataset, Dataset, callable]:
        """
        Returns:
        (train_dataset, eval_dataset, test_dataset)
            A tuple of `DatasetOfSequences` consisting of concatenated datasets
        """
        train_dataset, eval_datasets, test_datasets, transform = self.sequences()

        def __swap(dataset):
            if dataset[0] is not None:
                return ConcatDataset(dataset[0])
            return None

        train_dataset = __swap(train_dataset)
        eval_datasets = __swap(eval_datasets)
        test_datasets = __swap(test_datasets)

        return train_dataset, eval_datasets, test_datasets, transform

    def get_sequence_dataset(self) -> Tuple[Optional[DatasetOfSequences],
                                            Optional[DatasetOfSequences],
                                            Optional[DatasetOfSequences]]:
        """
        Returns:
            (train_dataset, eval_dataset, test_dataset) : A tuple of `DatasetOfSequences`
        """
        sequence_len = self.config.sequence_len
        train_dataset, eval_datasets, test_datasets, transform = self.sequences()
        # -- ME --
        # `train_dataset` is a (list(KITTI360Sequence), list(sequence_ids)) 

        def __to_sequence_dataset(dataset_pair):
            # -- ME --
            # `dataset_pair` is (list(KITTI360Sequence), list(sequence_ids))
            if dataset_pair is None or dataset_pair[0] is None:
                return None
            return DatasetOfSequences(sequence_len, dataset_pair[0], dataset_pair[1], transform=transform, num_points=self.config.num_points)

        return tuple(map(__to_sequence_dataset, [train_dataset, eval_datasets, test_datasets]))

    @abstractmethod
    def get_ground_truth(self, sequence_name):
        """Returns the ground truth for the dataset_config for a given sequence"""
        return None
