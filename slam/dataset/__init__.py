from enum import Enum

import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from pyLiDAR_SLAM.slam.common.utils import ObjectLoaderEnum
from pyLiDAR_SLAM.slam.dataset.configuration import DatasetLoader, DatasetConfig
from pyLiDAR_SLAM.slam.dataset.kitti_dataset import KITTIDatasetLoader, KITTIConfig
from pyLiDAR_SLAM.slam.dataset.nclt_dataset import NCLTDatasetLoader, NCLTConfig
from pyLiDAR_SLAM.slam.dataset.ford_dataset import FordCampusDatasetLoader, FordCampusConfig
from pyLiDAR_SLAM.slam.dataset.nhcd_dataset import NHCDDatasetLoader, NHCDConfig

from pyLiDAR_SLAM.slam.dataset.kitti_odometry_dataset import KittiOdometryConfig, KittiOdometryDataset
from pyLiDAR_SLAM.slam.dataset.kitti_360_dataset_2 import KITTI360Config, Kitti360Dataset
from pyLiDAR_SLAM.slam.dataset.kitti_360_bboxes_dataset import Kitti360BBoxesDataset, KITTI360BBoxesConfig
from pyLiDAR_SLAM.slam.dataset.kitti_360_captions_dataset import Kitti360CaptionsDataset, KITTI360CaptionsConfig
from pyLiDAR_SLAM.slam.dataset.VoteNet.kitti_vote_dataset import KittiVoteDataset, KITTIVoteConfig

from pyLiDAR_SLAM.slam.dataset.rosbag_dataset import _with_rosbag
from pyLiDAR_SLAM.slam.dataset.ct_icp_dataset import _with_ct_icp


class DATASET(ObjectLoaderEnum, Enum):
    """
    The different datasets covered by the dataset_config configuration
    A configuration must have the field dataset_config pointing to one of these keys
    """

    kitti = (KITTIDatasetLoader, KITTIConfig)
    kitti_odometry = (KittiOdometryDataset, KittiOdometryConfig)
    kitti_vote = (KittiVoteDataset, KITTIVoteConfig)
    
    kitti_360 = (Kitti360Dataset, KITTI360Config)
    kitti_360_bboxes = (Kitti360BBoxesDataset, KITTI360BBoxesConfig)
    kitti_360_captions = (Kitti360CaptionsDataset, KITTI360CaptionsConfig)
    
    nclt = (NCLTDatasetLoader, NCLTConfig)
    ford_campus = (FordCampusDatasetLoader, FordCampusConfig)
    nhcd = (NHCDDatasetLoader, NHCDConfig)
    if _with_rosbag:
        from slam.dataset.rosbag_dataset import RosbagDatasetConfiguration, RosbagConfig
        from slam.dataset.urban_loco_dataset import UrbanLocoConfig, UrbanLocoDatasetLoader
        rosbag = (RosbagDatasetConfiguration, RosbagConfig)
        urban_loco = (UrbanLocoDatasetLoader, UrbanLocoConfig)

    if _with_ct_icp:
        from slam.dataset.ct_icp_dataset import CT_ICPDatasetLoader, CT_ICPDatasetConfig
        ct_icp = (CT_ICPDatasetLoader, CT_ICPDatasetConfig)

    @classmethod
    def type_name(cls):
        return "dataset"
