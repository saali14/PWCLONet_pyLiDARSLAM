import logging
import logging.handlers
import numpy as np
import os
import sys
from pathlib import Path
import time

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.conf import dataclass, MISSING, field
from typing import Optional


from slam.eval.eval_odometry import OdometryResults


class LoggerConfig:
    """The configuration dataclass"""

    # --------------------------------
    # Default parameters
    odometry_log_file: str = '.output/backend.log'
    backend_log_file: str = '.output/odometry.log'
    eval_log_dir: str = '.output'
    err_log_file: str = '.output/err.log'

    odometry_level: str = 'INFO'
    backend_level: str = 'INFO'
    eval_level: str = 'INFO'
    err_level: str = 'ERROR'

    # ----------------
    # Debug parameters
    viz: bool = True
    viz_num_pointclouds: int = 200
    debug: bool = True


# -------------
# HYDRA Feature
# Automatically casts the config as a LoggerConfig object, and raises errors if it cannot do so
cs = ConfigStore.instance()
cs.store(name="logger_config", node=LoggerConfig)


class PoseLoggerFactory(object):

    _LOGS = {}

    def __init__(self, log_dir:str):
        self.log_dir = log_dir
        self.log_level = "INFO"
        self.log_name = "pose"
        self.elapsed_time = {}
        self.nb_processed_frames = {}
        self.odo_results = OdometryResults(self.log_dir)
        

    def init_sequence(self, sequence:int):
        log_file = str(Path(self.log_file) / str(sequence))
        PoseLoggerFactory._LOGS[sequence] = LoggerFactory.__create_logger(log_file, self.log_level, self.log_name)
        self.elapsed_time[sequence] = time.now()
        self.nb_processed_frames[sequence] = 0        

    
    def evaluate(self, sequence_name: str,
                frame: int,
                trajectory: np.ndarray,
                ground_truth: Optional[np.ndarray],
                elapsed: Optional[float] = None,
                last_frame: Optional[bool] = False):
        """Saves metrics and trajectory in a folder on disk"""

        self.odo_results.add_frame(sequence_name,
                              frame,
                              trajectory,
                              ground_truth,
                              elapsed)
        if last_frame:
            self.odo_results.close()


class LoggerFactory(object):

    _ODOMETRY_LOG = None 
    _BACKEND_LOG = None
    _EVAL_LOG = None
    _ERR_LOG = None

    _ELAPSED_TIME = {}

    @staticmethod
    def init(config: LoggerConfig):
        LoggerFactory._ODOMETRY_LOG = LoggerFactory.__create_logger(config.odometry_log_file, config.odometry_level, 'odometry')
        LoggerFactory._BACKEND_LOG = LoggerFactory.__create_logger(config.backend_log_file, config.backend_level, 'backend')
        LoggerFactory._ERR_LOG = LoggerFactory.__create_logger(config.err_log_file, config.err_log_file, 'error')

        LoggerFactory._EVAL_LOG = PoseLogger(config.eval_log_dir)
    
    def __create_logger(log_file, log_level, log_name):
        """
        A private method that interacts with the python
        logging module
        """
        # set the logging format
        log_format = "[%(name)s - %(process)d] %(asctime)s - %(levelname)s: %(message)s"

        handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", log_file), mode='w')
        formatter = logging.Formatter(log_format) # logging.BASIC_FORMAT
        handler.setFormatter(formatter)
        
        # Initialize the class variable with logger object
        LoggerFactory._LOG = logging.getLogger(log_name)
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%d-%m-%Y %H:%M:%S") # filename=log_file, filemode='w'
        LoggerFactory._LOG.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        LoggerFactory._LOG.addHandler(handler)
        
        # set the logging level based on the user selection
        if log_level == "INFO":
            LoggerFactory._LOG.setLevel(logging.INFO)
        elif log_level == "ERROR":
            LoggerFactory._LOG.setLevel(logging.ERROR)
        elif log_level == "DEBUG":
            LoggerFactory._LOG.setLevel(logging.DEBUG)
        elif log_level == "WARNING":
            LoggerFactory._LOG.setLevel(logging.WARNING)

        return LoggerFactory._LOG


    @staticmethod
    def get_logger(log_file, log_level):
        """
        A static method called by other modules to initialize logger in
        their own module
        """
        logger = LoggerFactory.__create_logger(log_file, log_level)
        
        # return the logger object
        return logger
    


if __name__ == "__main__":
    # initialize the logger object

    LoggerFactory.init()
    LoggerFactory._EVAL_LOG.info(" Test 1")
    LoggerFactory._ERR_LOG.info(" Test 2")
    #logger = LoggerFactory.get_logger("mymodule.py", log_level="INFO")
    #logger.info(" Inside module 1")
    #logger.warning(" Inside module 1")

    exit()