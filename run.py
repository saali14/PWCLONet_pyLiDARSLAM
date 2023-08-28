# Hydra and OmegaConf
import hydra
from omegaconf import DictConfig

# Project Imports
from slam.odometry.odometry_runner import SLAMRunner, SLAMRunnerConfig

import os
# Set environment variables
LD_LIBRARY_PATH = os.getenv('LD_LIBRARY_PATH')
CONDA_PREFIX = os.getenv('CONDA_PREFIX')
os.environ['LD_LIBRARY_PATH'] = f'{LD_LIBRARY_PATH}:{CONDA_PREFIX}/lib'


@hydra.main(config_path="config", config_name="slam")
def run_slam(cfg: DictConfig) -> None:
    """The main entry point to the script running the SLAM"""
    # -- ME -- sequences you want to run odometry on
    sequences = [3]
    _odometry_runner = SLAMRunner(SLAMRunnerConfig(**cfg))
    _odometry_runner.run_odometry(sequences=sequences)#, max_num_frames=7000)


if __name__ == "__main__":
    #import torch
    #torch.cuda.set_device("cuda:1")
    #torch.set_num_threads(5)
    run_slam()
