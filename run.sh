# HYDRA_FULL_ERROR=1 ./run.sh

INITIALIZATION="CV"
LOCAL_MAP="P"
ODOMETRY_LOCAL_MAP="kdtree"
if [ "$LOCAL_MAP" = "P" ]
then
    ODOMETRY_LOCAL_MAP="projective"
fi

export KITTI_360_ROOT=/data/3d_cluster/ADAS_data/KITTI-360/heavy_data       # The path to KITTI odometry benchmark files


export JOB_NAME=kitti_360_${INITIALIZATION}_${LOCAL_MAP}F2M                 # The folder to log hydra output files
export DATASET=kitti_360                                                    # Name of the Dataset to construct the destination folder 

# -m cProfile -o cprofileoutput.txt
python run.py slam/initialization=${INITIALIZATION} \
    slam/preprocessing=grid_sample \
    +slam/odometry/local_map=${ODOMETRY_LOCAL_MAP} \
    slam.odometry.local_map.local_map_size=20 \
    +slam/odometry/alignment=point_to_plane_GN \
    slam.odometry.max_num_alignments=15 \
    slam.odometry.alignment.gauss_newton_config.scheme=neighborhood \
    slam.odometry.alignment.gauss_newton_config.sigma=0.2 \
    dataset=kitti_360 \
    device=cuda:2 \
    num_workers=2 \
    slam.odometry.data_key=vertex_map \
    slam.odometry.viz_debug=False \
    hydra.run.dir=outputs/test/${JOB_NAME}


#INITIALIZATION="CV"
#LOCAL_MAP="Kd"
#ODOMETRY_LOCAL_MAP="kdtree"
#if [ "$LOCAL_MAP" = "P" ]
#then
#    ODOMETRY_LOCAL_MAP="projective"
#fi
#
#export JOB_NAME=kitti_360_${INITIALIZATION}_${LOCAL_MAP}F2M                                        The folder to log hydra output files
#
#python run.py slam/initialization=${INITIALIZATION} \
#    slam/preprocessing=grid_sample \
#    +slam/odometry/local_map=${ODOMETRY_LOCAL_MAP} \
#    slam.odometry.local_map.local_map_size=30 \
#    +slam/odometry/alignment=point_to_plane_GN \
#    slam.odometry.max_num_alignments=20 \
#    slam.odometry.alignment.gauss_newton_config.scheme=neighborhood \
#    slam.odometry.alignment.gauss_newton_config.sigma=0.2 \
#    dataset=kitti_360 \
#    device=cpu \
#    num_workers=1 \
#    slam.odometry.data_key=input_data \
#    slam.odometry.viz_debug=False \
#    hydra.run.dir=outputs/${JOB_NAME}



#+dataset.train_sequences=["0", "3", "4", "5", "6", "7", "9", "10"] \
#python run.py -m num_workers=8 device=cuda:1 \
#    slam/initialization=${INITIALIZATION} \
#    slam/odometry=icp_odometry \
#    +slam/odometry/alignment=point_to_plane_GN \
#    slam.odometry.max_num_alignments=15 \
#    slam.odometry.alignment.gauss_newton_config.scheme=neighborhood \
#    slam.odometry.alignment.gauss_newton_config.sigma=0.2 \
#    +slam/odometry/local_map=${ODOMETRY_LOCAL_MAP} \
#    slam.odometry.local_map.local_map_size=20 \
#    dataset=kitti_360 \
#    device=cuda:1 \
#    slam.odometry.viz_debug=False \
#    hydra.run.dir=.outputs/${JOB_NAME} \
#    slam.odometry.data_key=vertex_map
    


#python run.py num_workers=1 \
#    slam/initialization=CV \
#    slam/preprocessing=grid_sample \
#    slam/odometry=icp_odometry \
#    slam.odometry.viz_debug=True \
#    dataset=kitti_360 \
#    hydra.run.dir=.outputs/kitti360_CV

#python run.py num_workers=1 \
#    slam/initialization=CV \
#    slam/preprocessing=grid_sample \
#    slam/odometry=icp_odometry \
#    slam.odometry.viz_debug=True \
#    dataset=kitti_360 \
#    hydra.run.dir=.outputs/kitti360