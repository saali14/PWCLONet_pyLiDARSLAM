import torch
import numpy as np

import numpy as np
from scipy.spatial.transform.rotation import Rotation as R, Slerp
from scipy.interpolate.interpolate import interp1d

import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from pyLiDAR_SLAM.slam.common.utils import assert_debug, check_tensor
from pyLiDAR_SLAM.slam.common.rotation import torch_euler_to_mat, torch_mat_to_euler, torch_pose_matrix_jacobian_euler

from pyquaternion import Quaternion
import math


class PosesInterpolator:
    """Object which performs interpolation of poses using timestamps

    Poses and corresponding key timestamps are passed to the constructor.
    The PosesInterpolator returns a linear interpolation on these poses
    When called with new timestamps.
    """

    def __init__(self, poses: np.ndarray, timestamps: np.ndarray):
        check_tensor(poses, [-1, 4, 4], np.ndarray)
        check_tensor(timestamps, [-1], np.ndarray)
        self.min_timestamp = timestamps.min()
        self.max_timestamp = timestamps.max()

        # Slerp: Spherical Linear Interpolation of Rotations
        self.slerp = Slerp(timestamps, R.from_matrix(poses[:, :3, :3]))
        # returns a function whose call method uses interpolation to find the value of new points
        self.interp_tr = interp1d(timestamps, poses[:, :3, 3], axis=0)

    def __call__(self, timestamps: np.ndarray):
        if timestamps.min() < self.min_timestamp or timestamps.max() > self.max_timestamp:
            # limit the values in an array
            timestamps = np.clip(timestamps, self.min_timestamp, self.max_timestamp)
        tr = self.interp_tr(timestamps)
        rots = self.slerp(timestamps)

        poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(timestamps.shape[0], axis=0)
        poses[:, :3, :3] = rots.as_matrix()
        poses[:, :3, 3] = tr
        return poses


def transform_pointcloud(pointcloud: np.ndarray, tr: np.ndarray):
    """
    Applies the transform `tr` to the pointcloud

    Parameters
    ----------
    pointcloud : np.ndarray (N, 3)
    tr : np.ndarray (4, 4)
    """
    return np.einsum("ij,nj->ni", tr[:3, :3], pointcloud) + tr[:3, 3].reshape(1, 3)


class Pose(object):
    """
    A Pose is a tool to interpret tensors of float as SE3 poses

    Parameters
    ----------
    config : dict
        A dictionary with the configuration of the pose
    """

    def __init__(self, pose_type: str):
        self.pose_type = pose_type
        assert_debug(self.pose_type in self.__supported_poses())

    @staticmethod
    def __supported_poses():
        return ["euler", "quaternions"]  # TODO , "quaternions"

    def euler_convention(self):
        """
        Returns the euler convention used for the parametrisation of the rotation

        Fails if self.pose_type is not equal to "euler"
        """
        assert_debug(self.pose_type == "euler")
        return "xyz"

    def num_rot_params(self) -> int:
        """
        Returns
        -------
        int :
            The number of parameters of rotation for this representation
        """
        if self.pose_type == "quaternions":
            return 4
        else:
            return 3

    def num_params(self) -> int:
        """

        Returns
        -------
        int :
            The number of parameters (rotation + translation) for this representation
        """
        return self.num_rot_params() + 3

    def inverse_pose_matrix(self, params_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns the inverse of the pose matrix

        Parameters
        ----------
        params_tensor : [B, 6/7] or [B, 4, 4]
        """
        if len(params_tensor.shape) == 2:
            params_tensor = self.build_pose_matrix(params_tensor)
        check_tensor(params_tensor, [-1, 4, 4])

        inverse = torch.zeros_like(params_tensor)
        rt = params_tensor[:, :3, :3].permute(0, 2, 1)
        inverse[:, :3, :3] = rt
        inverse[:, :3, 3] = - torch.einsum("bij,bj->bi", rt, params_tensor[:, :3, 3])
        inverse[:, 3, 3] = 1.0
        return inverse

    def build_pose_matrix(self, params_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns a pose matrix tensor from a pose parameters tensor

        Parameters
        ----------
        params_tensor : torch.Tensor
            The tensor of the 6 or 7 parameters of the pose

        Returns
        -------
        torch.Tensor
            The tensor of matrix
        """
        check_tensor(params_tensor, [-1, self.num_rot_params() + 3])
        b = params_tensor.size(0)
        rotation_tensor = self.rot_matrix_from_params(params_tensor[:, 3:])
        pose = torch.cat([rotation_tensor, torch.zeros(b, 1, 3,
                                                       device=params_tensor.device,
                                                       dtype=params_tensor.dtype)], dim=1)  # [B, 4, 3]
        trans = torch.cat([params_tensor[:, :3],
                           torch.ones(b, 1, device=params_tensor.device, dtype=params_tensor.dtype)], dim=1) \
            .unsqueeze(2)  # [B, 4, 1]
        pose = torch.cat([pose, trans], dim=2)  # [B, 4, 4]
        return pose

    def __to_pose_matrix(self, pose: torch.Tensor):
        if len(pose.shape) == 3 and pose.size(1) == 4 and pose.size(2) == 4:
            t_pose_matrix = pose
        else:
            check_tensor(pose, [-1, self.num_rot_params() + 3])
            t_pose_matrix = self.build_pose_matrix(pose)
        return t_pose_matrix

    def apply_rotation(self, tensor: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotation part of the pose on the point cloud or normal cloud

        Parameters
        ----------
        tensor : [B, N, 3]
            A point or normal cloud tensor
        pose : [B, 4, 4] or [B, P]
            A pose matrix or pose params tensor
        """
        t_pose_matrix = self.__to_pose_matrix(pose)
        transformed = torch.einsum("bij,bnj->bni", t_pose_matrix[:, :3, :3], tensor)
        return transformed

    def apply_transformation(self, points_3d: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Applies a transformation to a point cloud

        Parameters
        ----------
        points_3d : [B, N, 3]
            A X, Y, Z point cloud tensor
        pose : [B, 4, 4] or [B, P]
            A pose matrix tensor or a pose params tensor
        """
        t_pose_matrix = self.__to_pose_matrix(pose)

        rot_matrix_t = t_pose_matrix[:, :3, :3].permute(0, 2, 1)
        points_3d = torch.matmul(points_3d, rot_matrix_t)
        tr = t_pose_matrix[:, :3, 3].unsqueeze(1)
        points_3d = points_3d + tr
        return points_3d

    def from_pose_matrix(self, pose_matrix_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns the tensor of the parameters of the pose

        Parameters
        ----------
        pose_matrix_tensor : torch.Tensor
            The matrix tensor [B, 4, 4]

        Returns
        -------
        torch.Tensor : [B, P]
            The pose parameters tensor.
            P is the degrees of freedom 6, (or 7 for 'quaternions')

        """
        rotation_matrix = pose_matrix_tensor[:, :3, :3]
        rot_params = self.rot_params_from_matrix(rotation_matrix)
        trans_params = pose_matrix_tensor[:, :3, 3]
        return torch.cat([trans_params, rot_params], dim=1)

    def rot_matrix_from_params(self, rot_params: torch.Tensor) -> torch.Tensor:
        """
        Builds a pose matrix tensor from its rotation parameters

        Parameters
        ----------
        rot_params : [B, ROT_P]
            The rotation parameters
        """
        if self.pose_type == "euler":
            return torch_euler_to_mat(rot_params, convention=self.euler_convention())
            # return TF3d.euler_angles_to_matrix(rot_params, convention=self.euler_convention())
        elif self.pose_type in ["quaternions", "quaternions_vec"]:
            quaternions = rot_params
            if self.pose_type == "quaternions_vec":
                # Transform the vector part of the quaternion (qx, qy, qz) into a unit quaternion
                quaternions = torch.cat([quaternions[:, :1].detach() * 0 + 1, quaternions], dim=1)

            # transform to unit quaternions
            norm_quat = quaternions / quaternions.norm(p=2, dim=1, keepdim=True)
            w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

            B = norm_quat.size(0)

            w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z

            rotation_matrix = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                                           2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                                           2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
            return rotation_matrix
        else:
            raise ValueError("Unrecognised pose type")


    def matrix_to_quaternion(self, rot_matrix: torch.Tensor) -> torch.Tensor:
        quaternions = np.zeros((rot_matrix.size(0), 4))
        for i in range(rot_matrix.size(0)):
            m00 = rot_matrix[i, 0, 0]
            m01 = rot_matrix[i, 0, 1]
            m02 = rot_matrix[i, 0, 2]
            m10 = rot_matrix[i, 1, 0]
            m11 = rot_matrix[i, 1, 1]
            m12 = rot_matrix[i, 1, 2]
            m20 = rot_matrix[i, 2, 0]
            m21 = rot_matrix[i, 2, 1]
            m22 = rot_matrix[i, 2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                            [m01+m10,     m11-m00-m22, 0.0,         0.0],
                            [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                            [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
            if q[0] < 0.0:
                np.negative(q, q)
            
            quaternions[i,:] = q
        return torch.from_numpy(q)
    
        quaternions = np.zeros((rot_matrix.size(0), 4))
        for i in range(rot_matrix.size(0)):
            quaternion = Quaternion(matrix=rot_matrix[i,:,:])
            quaternions[i,:] = quaternion.elements
        return torch.Tensor(quaternions).cuda()
    
    # https://github.com/zhenpeiyang/pytorch-learned-correspondence/blob/master/transformations.py
    def quaternion_from_matrix(matrix, isprecise=False):
        """Return quaternion from rotation matrix.
        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used.
        >>> q = quaternion_from_matrix(numpy.identity(4), True)
        >>> numpy.allclose(q, [1, 0, 0, 0])
        True
        >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
        >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
        True
        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R, True)
        >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
        True
        >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
        ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
        True
        >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
        ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
        True
        >>> R = random_rotation_matrix()
        >>> q = quaternion_from_matrix(R)
        >>> is_same_transform(R, quaternion_matrix(q))
        True
        >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
        >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
        ...                quaternion_from_matrix(R, isprecise=True))
        True
        """
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4, ))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 1, 2, 3
                if M[1, 1] > M[0, 0]:
                    i, j, k = 2, 3, 1
                if M[2, 2] > M[i, i]:
                    i, j, k = 3, 1, 2
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                            [m01+m10,     m11-m00-m22, 0.0,         0.0],
                            [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                            [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q


    def rot_params_from_matrix(self, rot_matrix: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            A [B, P] tensor with the parameters of the representation of the rotation matrices
        """
        if self.pose_type == "euler":
            return torch_mat_to_euler(rot_matrix, convention=self.euler_convention())
            # return TF3d.matrix_to_euler_angles(rot_matrix, convention=self.euler_convention())
        elif self.pose_type in ["quaternions", "quaternions_vec"]:
            quaternions = self.matrix_to_quaternion(rot_matrix)
            return quaternions
            raise NotImplementedError("")

        #     # Deal with the sign ambiguity of the quaternions : force the first parameter qw to 1
        #     quaternions = quaternions / quaternions[:, 0:1]
        #     if self.pose_type == "quaternions":
        #         unit_quaternions = quaternions / quaternions.norm(p=2, dim=1, keepdim=True)
        #         return unit_quaternions
        #     else:
        #         # returns unscaled rotation parameters (supposing that qw = 1)
        #         # Useful for pose prediction
        #         return quaternions[:, 1:4]
        else:
            raise ValueError(f"Unexpected pose_type {self.pose_type}")

    def pose_matrix_jacobian(self, pose_params: torch.Tensor):
        assert_debug(self.pose_type == "euler", 'Only euler angles are supported for now')
        return torch_pose_matrix_jacobian_euler(pose_params)
