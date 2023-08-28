
import numpy as np
import open3d
from scipy.spatial.transform import Rotation


def inv_q(q, scalar_last: bool = False):
    """
    The Quaternion Inverse (q-1 = conjugate(q)/|q|2)
    """
    q_2 = np.sum(q*q) + 1e-10
    q_conjugate = q * np.array([1, -1, -1, -1])
    q_inv = q_conjugate / q_2

    return q_inv


def warp(xyz, q, t, scalar_last: bool = False):
    """
    warp 3d coordinates\n
    Inputs:
        * `xyz`:    [N, 3]
        * `q`:      [4]
        * `t`:      [3]\n
    Return:
        * `xyz_warped`: [N,3]
    """

    q_inv = inv_q(q)    # [4]

    xyz = xyz.reshape((-1,3))

    xyz_ = np.concatenate((np.zeros((xyz.shape[0], 1)), xyz), axis=-1) # [N, 4]
    xyz_warped = mul_q_point(q, xyz_) # [N,4]
    xyz_warped = mul_point_q(xyz_warped, q_inv) # [N,4]
    xyz_warped = xyz_warped[:, 1:] # [N,3] 
    xyz_warped = xyz_warped + t.reshape(1,3) # [N,3]

    return xyz_warped # [N,3]


def new_warp(xyz, q, delta_q, t, scalar_last: bool = False):
    """
    warp 3d coordinates\n
    Inputs:
        * `xyz`:    [N, 3]
        * `q`:      [4]
        * `t`:      [3]\n
    Return:
        * `xyz_warped`: [N,3]
    """

    q_inv = inv_q(q)    # [4]
    delta_q_inv = inv_q(delta_q)

    xyz = xyz.reshape((-1,3))

    xyz_ = np.concatenate((np.zeros((xyz.shape[0], 1)), xyz), axis=-1) # [N, 4]

    xyz_unwarped = mul_q_point(q_inv, xyz_) # [N,4]
    xyz_unwarped = mul_point_q(xyz_unwarped, q) # [N,4]

    xyz_warped = mul_q_point(delta_q, xyz_unwarped) # [N,4]
    xyz_warped = mul_point_q(xyz_warped, delta_q_inv) # [N,4]

    xyz_warped = xyz_warped[:, 1:] # [N,3] 
    xyz_warped = xyz + t.reshape(1,3) # [N,3]

    xyz_unwarped = xyz_unwarped[:, 1:]

    return xyz_warped, xyz_unwarped # [N,3]


def mul_point_q(points, q, scalar_last: bool = False):
    """
    multiplication between points and Quaternion\n
    Inputs:
        * `points`:   [N, 4]
        * `q`:        [4]\n
    Return:
        * `q_result`: [N, 4]
    """

    points = points.reshape((-1, 4))

    q_result_0 = points[:, 0] * q[0] - points[:, 1] * q[1] - points[:, 2] * q[2] - points[:, 3] * q[3]
    q_result_0 = q_result_0.reshape((-1, 1))
    
    q_result_1 = points[:, 0] * q[1] + points[:, 1] * q[0] + points[:, 2] * q[3] - points[:, 3] * q[2]
    q_result_1 = q_result_1.reshape((-1, 1))

    q_result_2 = points[:, 0] * q[2] - points[:, 1] * q[3] + points[:, 2] * q[0] + points[:, 3] * q[1]
    q_result_2 = q_result_2.reshape((-1, 1))

    q_result_3 = points[:, 0] * q[3] + points[:, 1] * q[2] - points[:, 2] * q[1] + points[:, 3] * q[0]
    q_result_3 = q_result_3.reshape((-1, 1))

    q_result = np.concatenate([q_result_0, q_result_1, q_result_2, q_result_3], axis=-1)   # [N, 4]

    return q_result


def mul_q_point(q, points, scalar_last: bool = False):
    """
    multiplication between Quaternion and points\n
    Inputs:
        * `q`:        [4]
        * `points`:   [4, N]\n
    Return:
        * `q_result`: [4, N]
    """

    points = points.reshape((-1, 4))

    q_result_0 = q[0] * points[:, 0] - q[1] * points[:, 1] - q[2] * points[:, 2] - q[3] * points[:, 3]
    q_result_0 = q_result_0.reshape((-1, 1))
    
    q_result_1 = q[0] * points[:, 1] + q[1] * points[:, 0] + q[2] * points[:, 3] - q[3] * points[:, 2]
    q_result_1 = q_result_1.reshape((-1, 1))

    q_result_2 = q[0] * points[:, 2] - q[1] * points[:, 3] + q[2] * points[:, 0] + q[3] * points[:, 1]
    q_result_2 = q_result_2.reshape((-1, 1))

    q_result_3 = q[0] * points[:, 3] + q[1] * points[:, 2] - q[2] * points[:, 1] + q[3] * points[:, 0]
    q_result_3 = q_result_3.reshape((-1, 1))

    q_result = np.concatenate([q_result_0, q_result_1, q_result_2, q_result_3], axis=-1)  # [N, 4]

    return q_result


coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()

points = np.zeros((30, 3))
colors = np.zeros((30, 3))

points[:10, 0] = np.arange(0, 3, 0.3)
colors[:10, :] = np.tile(np.array([1., 0., 0.]), (10, 1))

points[10:20, 1] = np.arange(0, 3, 0.3)
colors[10:20, :] = np.tile(np.array([0., 1., 0.]), (10, 1))

points[20:, 2] = np.arange(0, 3, 0.3)
colors[20:, :] = np.tile(np.array([0., 0., 1.]), (10, 1))

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points)
pcd.colors = open3d.utility.Vector3dVector(colors)

# ------------

t = np.array([10, 0, 0])
r = Rotation.from_euler('xyz', (0, 0, 90), degrees=True)
q = r.as_quat()

q_ = np.copy(q)
q_[0] = q[-1]
q_[1:] = q[:-1]

points_1 = warp(points, q_, t)

pcd_1 = open3d.geometry.PointCloud()
pcd_1.points = open3d.utility.Vector3dVector(points_1)
pcd_1.colors = open3d.utility.Vector3dVector(colors)

# -----------

R = r.as_matrix()

coordinate_frame_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
coordinate_frame_1 = coordinate_frame_1.rotate(R)
coordinate_frame_1 = coordinate_frame_1.translate(t)

# --------

delta_t = np.array([0, 10, 5])
delta_r = Rotation.from_euler('xyz', (45, 45, 0), degrees=True)
delta_q = delta_r.as_quat()

delta_q_ = np.copy(delta_q)
delta_q_[0] = delta_q[-1]
delta_q_[1:] = delta_q[:-1]

q_new = np.squeeze(mul_q_point(delta_q_, q_))
#t_new = np.squeeze(warp(t, q_new, delta_t))
t_new, t_unwarped = new_warp(t, q_, delta_q_, delta_t)
t_new = np.squeeze(t_new)
t_unwarped = np.squeeze(t_unwarped)

points_2 = warp(points, q_new, t_new)
pcd_2 = open3d.geometry.PointCloud()
pcd_2.points = open3d.utility.Vector3dVector(points_2)
pcd_2.colors = open3d.utility.Vector3dVector(colors)

points_3 = warp(points, q_new, t_unwarped)
pcd_3 = open3d.geometry.PointCloud()
pcd_3.points = open3d.utility.Vector3dVector(points_3)
pcd_3.colors = open3d.utility.Vector3dVector(colors)

# -----------

delta_R = delta_r.as_matrix()

coordinate_frame_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
coordinate_frame_2 = coordinate_frame_2.rotate(R)
coordinate_frame_2 = coordinate_frame_2.translate(t)
coordinate_frame_2 = coordinate_frame_2.rotate(delta_R)
coordinate_frame_2 = coordinate_frame_2.translate(delta_t)

open3d.visualization.draw_geometries([pcd, pcd_1, pcd_2, coordinate_frame, coordinate_frame_1, coordinate_frame_2])
