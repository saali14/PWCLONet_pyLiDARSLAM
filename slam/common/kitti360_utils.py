import numpy as np
import pandas as pd
from typing import Dict, Union, List
import math

# open3d
try:
    import open3d

    _with_o3d = True
except ImportError:
    _with_o3d = False

from scipy.spatial.transform.rotation import Rotation as R, Slerp
from scipy.interpolate import interp1d
import logging
from pathlib import Path

import os
import sys


class KITTI360_INFOS:
    VELODYNE_HEIGHT = 1.73 # meters
    CAMERA_HEIGHT = 1.65
    GPS_IMU_HEIGHT = 0.93
    WHEEL_AXIS_HEIGHT = 0.3

    VELODYNE_FREQ = 10 # frames per second

    CAR_WIDTH = 1.60
    CAR_LENGTH = 2.71
    CAR_COMPLETE_LENGTH = CAR_LENGTH * 2
    CAR_HEIGHT = VELODYNE_HEIGHT # + GPS_IMU_HEIGHT

    VELODYNE_AXIS = 'x,y,z'
    CAMERA_AXIS = 'z,-x,-y'
    GPS_IMU_AXIS = 'x,y,z'

    """GPU_TO_VELODYNE = np.array([[0.3581544,     -0.4794579,     -0.8011526,     0.81],
                                [-0.8011526,    -0.5984601,     0.0000000,      -0.32],
                                [-0.4794579,    0.6418455,      -0.5984601,     0.8],
                                [0.,            0.,             0.,             1.] ])"""
    
    GPU_TO_VELODYNE = np.array([[1.0000000,  0.,  0.,     0.81],
                                [0.0000000, -1.,  0.,     -0.32],
                                [0.0000000, -0., -1.,     0.8],
                                [0.,         0.,             0.,     1.] ])
    
    CAR_BBOX_LEFT_UP_FRONT      = np.array([CAR_LENGTH/2., CAR_WIDTH/2., 0.])
    CAR_BBOX_LEFT_DOWN_FRONT    = np.array([CAR_LENGTH/2., CAR_WIDTH/2., -CAR_HEIGHT])
    CAR_BBOX_RIGHT_UP_FRONT     = np.array([CAR_LENGTH/2., -CAR_WIDTH/2., 0.])
    CAR_BBOX_RIGHT_DOWN_FRONT   = np.array([CAR_LENGTH/2., -CAR_WIDTH/2., -CAR_HEIGHT])
    CAR_BBOX_LEFT_UP_BACK       = CAR_BBOX_LEFT_UP_FRONT - np.array([CAR_LENGTH, 0., 0.])
    CAR_BBOX_LEFT_DOWN_BACK     = CAR_BBOX_LEFT_DOWN_FRONT - np.array([CAR_LENGTH, 0., 0.])
    CAR_BBOX_RIGHT_UP_BACK      = CAR_BBOX_RIGHT_UP_FRONT - np.array([CAR_LENGTH, 0., 0.])
    CAR_BBOX_RIGHT_DOWN_BACK    = CAR_BBOX_RIGHT_DOWN_FRONT - np.array([CAR_LENGTH, 0., 0.])

    CAR_BBOX = np.concatenate((CAR_BBOX_LEFT_UP_FRONT.reshape(1,3), CAR_BBOX_LEFT_DOWN_FRONT.reshape(1,3), CAR_BBOX_RIGHT_UP_FRONT.reshape(1,3), CAR_BBOX_RIGHT_DOWN_FRONT.reshape(1,3), \
                               CAR_BBOX_LEFT_DOWN_BACK.reshape(1,3), CAR_BBOX_LEFT_UP_BACK.reshape(1,3), CAR_BBOX_RIGHT_DOWN_BACK.reshape(1,3), CAR_BBOX_RIGHT_UP_BACK.reshape(1,3)), axis=0)
    
    CAR_CENTER = np.array([CAR_LENGTH-(0.81 - 0.05), 0., -CAR_HEIGHT/2.])

    CAR_LINES = np.array([[0,1], [0,2], [1,3], [2,3], [4,5], [5,7], [6,7], [4,6], [0,5], [1,4], [2,7], [3,6]])

    CAR_TRIANGLES = np.array([[0, 1, 2], [0, 1, 5], [0, 5, 7], [0, 2, 7], [1, 2, 3], [1, 4, 5], [1, 3, 4], [2, 6, 7], [2, 3, 6], [3, 4, 6], [4, 6, 7], [4, 5, 7]])

    CAR_COLOR = np.array([1., 0., 0.])


class KITTI360_IO:

    CAM0_TO_POSE = np.array([[0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039],
                            [0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093],
                            [0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000],
                            [0, 0, 0, 1]], dtype=np.float64)

    VELO_TO_CAM0 = np.linalg.inv(np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                                            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                                            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                                            [0, 0, 0, 1]], dtype=np.float64))

    VELO_TO_POSE = CAM0_TO_POSE.dot(VELO_TO_CAM0)


    def df_to_poses(df: pd.DataFrame):
        
        """
        Converts a DataFrame [N, 12] to an array of pose matrices [N, 4, 4]
        """
        shape = df.shape
        assert(len(shape) == 2)
        assert(shape[1] == 12)
        nrows = shape[0]

        poses_array = df.values
        reshaped = np.tile(np.eye(4), nrows).reshape(nrows, 4, 4)
        reshaped[:, :3,:4] = poses_array[:, :].reshape((nrows, 3, 4))

        return reshaped


    def read_csv_poses(filePath: str):
        """
        reads a csv file containing poses and converts them to a numpu array of pose matrices [N, 4, 4]
        poses transform points from LiDAR to World coordinates
        """
        df = pd.read_csv(filePath, header=0, sep=',')
        poses = KITTI360_IO.df_to_poses(df)

        return poses


    def loadPoses(file_name: str, toCameraCoord: bool = False, sep: str =' ', ali: bool = False, velo_to_world: bool = False):
        '''
            Each line in the file should follow one of the following structures:
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        if ali:
            with open(file_name) as f:
                lines = f.readlines()
                poses = {}
                for i in range(len(lines) // 5):
                    t = []
                    for j in range(1, 5):
                        t.append(lines[i*5 + j].split())
                    poses[i] = np.array(t).astype(np.float32)
            return poses

        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(sep)]
            withIdx = int(len(line_split)==13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt

            if velo_to_world:
                P = np.matmul(P, KITTI360_IO.VELO_TO_POSE)

            if toCameraCoord:
                poses[frame_idx] = KITTI360_IO.to_camera_coord(P)
            else:
                poses[frame_idx] = P
        return poses


    def load_raw_3D_pcd(frame: int, raw3DPcdPath: str) -> np.ndarray:
        pcdFile = os.path.join(raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 


    def kitti_360_poses(file_path: str):
        """
        the rigid body transform from GPU/IMU coordinates to a world coordinate system
        """
        df = pd.read_csv(file_path, sep=" ", header=None)
        poses = df.values  # [N, 13] array

        frame_indices = poses[:, 0].astype(np.int32)
        pose_data = poses[:, 1:]

        n = pose_data.shape[0]
        pose_data = np.concatenate((pose_data, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
                                axis=1)
        poses = pose_data.reshape((n, 4, 4))  # [N, 4, 4]
        return frame_indices, poses


    def read_timestamps(file_path: str):
        """Read a timestamps file and convert it to float64 values
        """
        df = pd.read_csv(file_path, header=None, sep=",", names=["instants"],
                        dtype={"instants": "str"}, parse_dates=["instants"])
        timestamps = df.values.astype(np.int64).astype(np.float64)
        return timestamps.reshape(-1)


    def drive_foldername(drive_id: int):
        return f"2013_05_28_drive_{drive_id:04}_sync"


    def from_drive_foldername(drive_name: str):
        return int(drive_name.split('_')[-2])


    def window_name(window_start: int, window_end: int):
        return f'{str(window_start).zfill(10)}_{str(window_end).zfill(10)}'


    def window_from_file_name(file_name):
        file_name = os.path.splitext(file_name)[0]
        window = file_name.split('_')
        window_start = int(window[0])
        window_end = int(window[1])

        return window_start, window_end


    def get_sequence_poses(root_dir: str, drive_id: int, velo_to_world: bool = True, relative: bool = False):
        """
        Returns the absolute poses of a given drive
        From velodyne to world coordinate system
        """
        if not drive_id in [0, 2, 3, 4, 5, 6, 7, 9, 10]:
            raise RuntimeError(f'Unkown \'drive_id\': {drive_id}.\nKITTI-360 only accepts drive ids from [0, 2, 3, 4, 5, 6, 7, 9, 10].')
        
        sequence_foldername = KITTI360_IO.drive_foldername(drive_id)
        root_dir = Path(root_dir)
        velodyne_path = root_dir / "data_3d_raw" / sequence_foldername / "velodyne_points"
        timestamps_path = velodyne_path / "timestamps.txt"

        if not relative:
            gt_file = root_dir / "data_poses" / sequence_foldername / "poses.txt"
        else:
            gt_file = root_dir / "data_poses" / sequence_foldername / "poses_diff.txt"

        gt_poses = None
        if gt_file.exists():
            # absolute poses
            # the rigid body transform from GPU/IMU coordinates to a world coordinate system
            index_frames, poses = KITTI360_IO.kitti_360_poses(str(gt_file))
            timestamps = KITTI360_IO.read_timestamps(str(timestamps_path))

            poses_key_times = timestamps[index_frames]
            rotations = R.from_matrix(poses[:, :3, :3])
            slerp = Slerp(poses_key_times, rotations)

            # Clamp timestamps at key times to allow interpolation
            timestamps = timestamps.clip(min=poses_key_times.min(), max=poses_key_times.max())

            frame_orientations = slerp(timestamps)
            frame_translations = interp1d(poses_key_times, poses[:, :3, 3], axis=0)(timestamps)

            # Compute one pose per frame by interpolating of the ground truth (there is less than a frame per pose)
            gt_poses = np.zeros((timestamps.shape[0], 4, 4), dtype=np.float64)
            gt_poses[:, :3, :3] = frame_orientations.as_matrix()
            gt_poses[:, :3, 3] = frame_translations
            gt_poses[:, 3, 3] = 1.0

            if velo_to_world:
                # Convert poses to the poses in the frame of the lidar
                gt_poses = np.einsum("nij,jk->nik", gt_poses, KITTI360_IO.VELO_TO_POSE)
                # -- ME -- lidar coordinates to world coordinates

        else:
            logging.warning(f"[KITTI-360]The ground truth filepath {gt_file} does not exist")
        return gt_poses


    def save_poses(poses, file_path: str):
        if poses is None:
            raise RuntimeError('[save_poses]: poses is None')
        
        
        if isinstance(poses, dict):
            np_poses = np.zeros((len(poses), 3, 4))
            frames = np.zeros(len(poses), dtype=int)
            for i, frame in enumerate(sorted(poses.keys())):
                np_poses[i] = poses[frame][:3,:]
                frames[i] = frame

            #frames = frames.reshape(len(poses), 1)
            np_poses = np_poses.reshape((len(poses), 12))
            #np_poses = np.concatenate([frames, np_poses], axis=1)

        elif isinstance(poses, np.ndarray):
            np_poses = poses
            frames = np.arange(np_poses.shape[0])
            if len(np_poses.shape) == 3:
                if (np_poses.shape[1] != 4) or ((np_poses.shape[2] != 4)):
                    raise RuntimeError('[save_poses]: poses should be numpy array of size (-1, 4, 4)')
                np_poses = np_poses[:, :3, :].reshape((len(np_poses), 12))
            elif len(np_poses.shape) == 2:
                if (np_poses.shape[1] != 12) or (np_poses.shape[1] != 13):
                    raise RuntimeError('[save_poses]: poses should be numpy array of size (-1, 12|13)')
            else:
                raise RuntimeError('[save_poses]: poses should be numpy array of size (-1, 4, 4) or (-1, 12|13)')
            
            if poses.shape[1] == 13:
                frames = np_poses[:, 0]
                np_poses = np_poses[:, 1:]
        
        else:
            raise RuntimeError('[save_poses]: poses should be either dict or numpy array')
        
        df = pd.DataFrame(data=np_poses, index=frames)
        df.to_csv(file_path, sep=' ', header=False, index=True)

        #np.savetxt(file_path, np_poses, delimiter=' ')


    def loadWindowNpy(file_path: str):
        array = np.load(file_path).reshape(-1,8)
        points = array[:,:3].astype(np.float32)
        colors = array[:,3:6].astype(np.float32)
        semanticIds = array[:,6].astype(np.int32)
        instanceIds = array[:,7].astype(np.int32)

        return points, colors, semanticIds, instanceIds


class KITTI360_TRANSFORMATIONS:

    def transformPcd(pcd, pose: np.ndarray):
        """
        transforms open3d Point Cloud object using pose 
        """
        if not _with_o3d:
            logging.warning("Open3D (open3d python module) not found, some features will be disabled")
        assert(pose.shape[1] == 4 and pose.shape[0] == 4)
        return pcd.transfrom(pose)


    def transformPcdPoints(pcdPoints: np.ndarray, pose: np.ndarray):
        """
        transforms Point Cloud points using pose
        """
        assert(pcdPoints.shape[1] == 3)
        assert(pose.shape[0] == 4 and pose.shape[1] == 4)
        return (np.matmul(pose[:3, :3], pcdPoints.T)).T + pose[:3, 3]
        # return ((pose[:3, :3] @ pcdPoints.T).T + pose[:3, 3]).T


    def to_camera_coord(pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0,   0,   1,  0],
                            [-1,  0,   0,  0],
                            [0,  -1,   0,  0],
                            [0,   0,   0,  1]])
        inv_R_C2L = np.linalg.inv(R_C2L)            
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot
    

    def velo_to_pose(pose_mat_velo_to_world):
        if isinstance(pose_mat_velo_to_world, dict):
            frames = np.sort(list(pose_mat_velo_to_world.keys()))
            poses = np.array([pose_mat_velo_to_world[frame] for frame in frames]).reshape((len(frames), 4, 4))
            if len(frames) == 1:
                poses = np.squeeze(poses)
        elif isinstance(pose_mat_velo_to_world, np.ndarray):
            poses = np.copy(pose_mat_velo_to_world)
        else:
            raise RuntimeError(f'[velo_to_pose] poses should be either dict or numpy array and not {type(poses)}')

        if len(poses.shape) == 2:
            pose_mat_pose_to_world = poses @ np.linalg.inv(KITTI360_IO.VELO_TO_POSE)
        elif len(poses.shape) == 3:
            pose_mat_pose_to_world = np.einsum("nij,jk->nik", poses, np.linalg.inv(KITTI360_IO.VELO_TO_POSE))
        else:
            raise RuntimeError(f'[velo_to_pose] Unrecognized shape of poses {poses.shape}')
        
        if isinstance(pose_mat_velo_to_world, dict):
            return_pose_mat_pose_to_world = {}
            if len(frames) == 1:
                return_pose_mat_pose_to_world[frames[0]] = pose_mat_pose_to_world
            else:
                for i, frame in enumerate(frames):
                    return_pose_mat_pose_to_world[frame] = pose_mat_pose_to_world[i, :, :]
            return return_pose_mat_pose_to_world

        return pose_mat_pose_to_world


    def to3dPcd(points, colors=None):
        if not _with_o3d:
            logging.warning("Open3D (open3d python module) not found, some features will be disabled")
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = open3d.utility.Vector3dVector(np.tile(colors, (len(points),1)))
        return pcd
    

    def convert_to_relative(abs_poses, from_first_frame: bool = False):
        assert (len(abs_poses.shape) == 3) and (abs_poses.shape[1:] == (4,4)), '[KITTI360_TRANSFORMATIONS.convert_to_relative] Unsuported shape of abs_poses'
        
        abs_poses_shifted = KITTI360_TOOLS.shift_poses(abs_poses)
        relative_poses = np.linalg.inv(abs_poses) @ abs_poses_shifted #np.einsum("nij,njk->nik", np.linalg.inv(abs_poses), abs_poses_shifted)
        if from_first_frame:
            relative_poses[0,:,:] = np.eye(4)
        
        return relative_poses
    

    def convert_to_absolute(relative_poses, first_transformation: np.ndarray = None, velo_to_pose: bool = False):
        if first_transformation is None:
            previous_abs_pose = np.eye(4)
        else:
            previous_abs_pose = first_transformation
            
        if isinstance(relative_poses, np.ndarray):
            assert len(relative_poses.shape) == 3 and relative_poses.shape[1:] == (4,4), f'[KITTI360_TRANSFORMATIONS.convert_to_absolute] Unsuported shape of relative_poses {relative_poses.shape}'

            abs_poses = np.tile(np.eye(4), (len(relative_poses), 1, 1))
            for i in range(relative_poses.shape[0]):
                abs_poses[i,:,:] = relative_poses[i,:,:] @ previous_abs_pose
                previous_abs_pose = abs_poses[i,:,:]               

            abs_poses = np.linalg.inv(abs_poses)

        elif isinstance(relative_poses, dict):
            abs_poses = dict()
            for frame_id in sorted(relative_poses.keys()):
                abs_poses[frame_id] = np.linalg.inv(relative_poses[frame_id] @ np.linalg.inv(previous_abs_pose))
                previous_abs_pose = abs_poses[frame_id]

        if velo_to_pose:
            abs_poses = KITTI360_TRANSFORMATIONS.velo_to_pose(abs_poses)

        return abs_poses


class KITTI360_TOOLS:

    def trajectoryDistances(poses: Union[Dict[int, np.ndarray], np.ndarray]):
        '''
            Compute the cummulative length of the trajectory from the first frame
            poses: Dict[frame_idx: pose]
        '''
        if isinstance(poses, dict):
            dist = [0]
            sort_frame_idx = sorted(poses.keys())
            for i in range(len(sort_frame_idx)-1):
                cur_frame_idx = sort_frame_idx[i]
                next_frame_idx = sort_frame_idx[i+1]
                P1 = poses[cur_frame_idx]
                P2 = poses[next_frame_idx]
                dx = P1[0,3] - P2[0,3]
                dy = P1[1,3] - P2[1,3]
                dz = P1[2,3] - P2[2,3]
                dist.append(dist[i] + np.sqrt(dx**2+dy**2+dz**2))
        else:
            shifted_poses = KITTI360_TOOLS.shift_poses(poses)
            dists = np.linalg.norm(poses[:,:3,3] - shifted_poses[:,:3,3], axis=1)

            assert dists.shape[0] == poses.shape[0]

            dist = np.zeros(poses.shape[0])
            dist[1:] = np.cumsum(dists).tolist()[1:]

        return dist


    def lastFrameFromSegmentLength(dist: Union[np.ndarray, List], first_frame: int, len_: int):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1


    def shift_poses(poses: np.ndarray):
        """
            shifts the poses by one frame
        """
        shifted = poses[:-1, :4, :4] # remove the last pose
        shifted = np.concatenate([np.expand_dims(np.eye(4), axis=0), shifted], axis=0) # add I as a first pose
        return shifted


    def compute_cumulative_trajectory_length(trajectory: np.ndarray):
        """
            the cumulative distance traveled between each two consecutive frames
        """
        shifted = KITTI360_TOOLS.shift_poses(trajectory) # shifts the trajectory by one frame: (n, 4, 4) where n = len(trajectory)
        lengths = np.linalg.norm(shifted[:, :3, 3] - trajectory[:, :3, 3], axis=1) # lengths between each two consecutive frames (n,)
        lengths = np.cumsum(lengths) # np.cumsum([1, 2, 3]) = [1, 3, 6]
        return lengths

    
    def euler2mat(anglex, angley, anglez):

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        
        R_trans = Rx.dot(Ry).dot(Rz)

        return R_trans


    def mat2euler(M, cy_thresh=None, seq='zyx'):

        M = np.asarray(M)
        if cy_thresh is None:
            cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33*r33 + r23*r23)
        if seq=='zyx':
            if cy > cy_thresh: # cos(y) not close to zero, standard form
                z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else: # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21,  r22)
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = 0.0
        elif seq=='xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi/2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi/2
        else:
            raise Exception('Sequence not recognized')
        return z, y, x


    def euler2quat(z=0, y=0, x=0, isRadian=True):
        ''' Return quaternion corresponding to these Euler angles
        Uses the z, then y, then x convention above
        Parameters
        ----------
        z : scalar
            Rotation angle in radians around z-axis (performed first)
        y : scalar
            Rotation angle in radians around y-axis
        x : scalar
            Rotation angle in radians around x-axis (performed last)
        Returns
        -------
        quat : array shape (4,)
            Quaternion in w, x, y z (real, then vector) format
        Notes
        -----
        We can derive this formula in Sympy using:
        1. Formula giving quaternion corresponding to rotation of theta radians
            about arbitrary axis:
            http://mathworld.wolfram.com/EulerParameters.html
        2. Generated formulae from 1.) for quaternions corresponding to
            theta radians rotations about ``x, y, z`` axes
        3. Apply quaternion multiplication formula -
            http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
            formulae from 2.) to give formula for combined rotations.
        '''
    
        if not isRadian:
            z = ((np.pi)/180.) * z
            y = ((np.pi)/180.) * y
            x = ((np.pi)/180.) * x
        z = z/2.0
        y = y/2.0
        x = x/2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        return np.array([
                        cx*cy*cz - sx*sy*sz,
                        cx*sy*sz + cy*cz*sx,
                        cx*cz*sy - sx*cy*sz,
                        cx*cy*sz + sx*cz*sy])


    def quat2mat_batch(q, scalar_last=False):
        """
            q: w, x, y, z   if scalar_last is False
        """
        new_q = q
        if not scalar_last:
            new_q = KITTI360_TOOLS.switch_quat(q, scalar_last=True)
            """if len(q.shape) == 2:
                new_q = np.zeros((q.shape[0], q.shape[1]))
                new_q[:,:-1] = q[:,1:]
                new_q[:,-1] = q[:,0]
            elif len(q.shape) == 1:
                new_q = np.zeros((q.shape[0]))
                new_q[:-1] = q[1:]
                new_q[-1] = q[0]
            else:
                raise RuntimeError(f'[quat2mat_batch] Unrecognized shape of quaternions: {q.shape}')"""
        r = R.from_quat(new_q)  # x, y, z, w
        return r.as_matrix()


    def quat2mat(q):
    
        ''' Calculate rotation matrix corresponding to quaternion
        https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
        Parameters
        ----------
        q : 4 element array-like
        Returns
        -------
        M : (3,3) array
        Rotation matrix corresponding to input quaternion *q*
        Notes
        -----
        Rotation matrix applies to column vectors, and is applied to the
        left of coordinate vectors.  The algorithm here allows non-unit
        quaternions.
        References
        '''
        
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-8:
            return np.eye(3)
        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array(
            [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    

    def params2mat(params, scalar_last=False):
        t = params[:3]
        q = params[3:]

        if len(q) == 4:
            R = KITTI360_TOOLS.quat2mat_batch(q, scalar_last=scalar_last)
        elif len(q) == 3:
            R = KITTI360_TOOLS.euler2mat(q)
        else:
            raise RuntimeError('[KITTI_TOOLS.params2mat] the number of rotation parameters should be either 3 or 4')
        
        transformation = np.eye(4)
        transformation[:3,:3] = R
        transformation[:3,3] = t

        return transformation


    def mat2quat(m, scalar_last=False):
        rot = R.from_matrix(m)
        quat = rot.as_quat()
        
        if not scalar_last:
            return_quat = KITTI360_TOOLS.switch_quat(quat, scalar_last=False)
            """return_quat = np.zeros_like(quat)
            if len(return_quat.shape) == 2:
                return_quat[:,0] = quat[:,-1]
                return_quat[:,1:] = quat[:,:-1]
            elif len(return_quat.shape) == 1:
                return_quat[0] = quat[-1]
                return_quat[1:] = quat[:-1]
            else:
                raise RuntimeError(f'[mat2quat] Unrecognized shape of quaternions: {quat.shape}')"""
        
            return return_quat
        else:
            return quat


    def switch_quat(q, scalar_last:bool =False):
        """
            Set scalar_last to True if you want the scalar to be the last one
        """
        new_q = np.copy(q)
        if len(q.shape) == 2:
            if scalar_last:
                new_q[:,:-1] = q[:,1:]
                new_q[:,-1] = q[:,0]
            else:
                new_q[:,1:] = q[:,:-1]
                new_q[:,0] = q[:,-1]
        elif len(q.shape) == 1:
            if scalar_last:
                new_q[:-1] = q[1:]
                new_q[-1] = q[0]
            else:
                new_q[1:] = q[:-1]
                new_q[0] = q[-1]
        else:
            raise RuntimeError(f'[switch_quat] Unrecognized shape of quaternions: {q.shape}')

        return new_q
