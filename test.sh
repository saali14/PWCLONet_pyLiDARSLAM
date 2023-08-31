# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 ./test.sh

# ------------- ENVIRONMENT VARIABLES -------------
export DATASET=kitti_odometry
export JOB_NAME=test_pwclonet
export TRAIN_DIR=./test       # Path to the output models

export PYLIDAR_SLAM_PWCLONET_ABS_PATH=/mnt/isilon/melamine/pylidar-slam-pwclonet
export KITTI360_DATASET=/mnt/isilon/melamine/KITTI-360/heavy_data
export KITTI_DATASET=/mnt/isilon/melamine/KITTI/dataset

# ------------- Input checkpoint file -------------
export IN_CHECPOINT_FILE=/mnt/isilon/melamine/pylidar-slam-pwclonet/Best_Models/mine_kitti_shuffling_e_4_trans_pap/train/train/checkpoints/best.ckp

# Launches the Training of PoseNet
#dataset.num_points=8192 -m pdb 
python train.py +device=cuda:0 +num_workers=16 +num_epochs=30 dataset=${DATASET} +do_train=False +do_test=True +batch_size=4 +in_checkpoint_file=${IN_CHECPOINT_FILE}
