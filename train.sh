
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 ./train.sh

export DATASET=kitti_odometry
export JOB_NAME=train_pwclonet_mine_kitti_odometry
export TRAIN_DIR=./train       # Path to the output models 

export PYLIDAR_SLAM_PWCLONET_ABS_PATH=/mnt/isilon/melamine/pylidar-slam-pwclonet

# Launches the Training of PoseNet
#dataset.num_points=8192 -m pdb 
python train.py +device=cuda:0 +num_workers=16 +num_epochs=120 dataset=${DATASET} +do_train=True +do_test=False +batch_size=8