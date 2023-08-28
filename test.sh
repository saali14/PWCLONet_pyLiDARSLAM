
# CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 ./test.sh

export DATASET=kitti_odometry
export JOB_NAME=train_pwclonet_mine_odometry_evaluate
export TRAIN_DIR=./train       # Path to the output models 

export IN_CHECPOINT_FILE=/mnt/isilon/melamine/relidar-slam/pyLiDAR_SLAM/Best_Models/mine_kitti_shuffling_e_4_trans_pap/train/checkpoints/best.ckp

# Launches the Training of PoseNet
#dataset.num_points=8192 -m pdb 
python train.py +device=cuda:2 +num_workers=16 +num_epochs=30 dataset=${DATASET} +do_train=False +do_test=True +batch_size=8 +in_checkpoint_file=${IN_CHECPOINT_FILE}