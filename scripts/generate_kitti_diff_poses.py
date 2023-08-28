
import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from kitti360Scripts.kitti360scripts.helpers.common import KITTI360_IO, KITTI360_TRANSFORMATIONS


def generate_kitti_diff_poses(datapath):

    poses_dir_path = os.path.join(datapath, 'data_poses')
    sequences_dirs = os.listdir(poses_dir_path)
    for seq_dir in sequences_dirs:
        seq_dir_path = os.path.join(poses_dir_path, seq_dir)
        if not os.path.isdir(seq_dir_path):
            continue

        #seq = KITTI360_IO.from_drive_foldername(sequences_dirs)
        seq = int(seq_dir.split('_')[-2])

        if seq not in [0, 2, 3, 4, 5, 6, 7, 9, 10]:
            continue

        abs_poses = KITTI360_IO.get_sequence_poses(datapath, seq, velo_to_world=True)
        rel_poses = KITTI360_TRANSFORMATIONS.convert_to_relative(abs_poses, from_first_frame=True)

        save_path = os.path.join(seq_dir_path, 'poses_diff.txt')
        KITTI360_IO.save_poses(rel_poses, save_path)

        print(f'Sequence {seq} done')


if __name__ == '__main__':
    datapath = '/mnt/isilon/melamine/KITTI/heavy_data'
    generate_kitti_diff_poses(datapath)

    print('All sequences were processed successfuly')