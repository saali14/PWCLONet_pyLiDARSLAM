
import torch
import numpy as np


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


def inv_q(q, device, scalar_last: bool = False):
    """
    The Quaternion Inverse (q-1 = conjugate(q)/|q|2)
    """
    q_2 = torch.sum(q*q, dim=-1, keepdim=True) + 1e-10
    q_conjugate = q * torch.tile(torch.tensor([1, -1, -1, -1]).to(device), [q.size(0), 1])
    q_inv = q_conjugate / q_2

    return q_inv


def warp(xyz, q, t, device, scalar_last: bool = False):
    """
    warp 3d coordinates\n
    Inputs:
        * `xyz`:    [B, 3, N]
        * `q`:      [B, 4, 1]
        * `t`:      [B, 3, 1]\n
    Return:
        * `xyz_warped`: [B, 3, N]
    """
    batch_size, _, num_points = xyz.size()

    q_ = torch.squeeze(q, dim=2)        # [B, 4]
    q_inv = inv_q(q_, device=device)    # [B, 4]

    xyz_ = torch.cat((torch.zeros([batch_size, 1, num_points]).to(device), xyz), dim=1) # [B, 4, N]
    xyz_warped = mul_q_point(q, xyz_) # [B, 4, N]
    xyz_warped = mul_point_q(xyz_warped, q_inv) # [B, 4, N]
    xyz_warped = xyz_warped[:, 1:, :] # [B, 3, N] 
    xyz_warped = xyz_warped + t # [B, 3, N]

    return xyz_warped # [B, 3, N]


def mul_point_q(points, q, scalar_last: bool = False):
    """
    multiplication between points and Quaternion\n
    Inputs:
        * `points`:   [B, 4, N]
        * `q`:        [B, 4, 1] or [B, 4]\n
    Return:
        * `q_result`: [B, 4, N]
    """

    batch_size = points.size(0)
    q = torch.reshape(q, [batch_size, 4, 1])

    q_t = torch.permute(q, (0, 2, 1)).contiguous()  # [B, 1, 4]
    points_t = torch.permute(points, (0, 2, 1)).contiguous()  # [B, N, 4]

    q_result_0 = torch.multiply(points_t[ :, :, 0], q_t[ :, :, 0])-torch.multiply(points_t[ :, :, 1], q_t[ :, :, 1])-torch.multiply(points_t[ :, :, 2], q_t[ :, :, 2])-torch.multiply(points_t[ :, :, 3], q_t[ :, :, 3])
    q_result_0 = torch.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = torch.multiply(points_t[ :, :, 0], q_t[ :, :, 1])+torch.multiply(points_t[ :, :, 1], q_t[ :, :, 0])+torch.multiply(points_t[ :, :, 2], q_t[ :, :, 3])-torch.multiply(points_t[ :, :, 3], q_t[ :, :, 2])
    q_result_1 = torch.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = torch.multiply(points_t[ :, :, 0], q_t[ :, :, 2])-torch.multiply(points_t[ :, :, 1], q_t[ :, :, 3])+torch.multiply(points_t[ :, :, 2], q_t[ :, :, 0])+torch.multiply(points_t[ :, :, 3], q_t[ :, :, 1])
    q_result_2 = torch.reshape(q_result_2, [batch_size, -1, 1])

    q_result_3 = torch.multiply(points_t[ :, :, 0], q_t[ :, :, 3])+torch.multiply(points_t[ :, :, 1], q_t[ :, :, 2])-torch.multiply(points_t[ :, :, 2], q_t[ :, :, 1])+torch.multiply(points_t[ :, :, 3], q_t[ :, :, 0])
    q_result_3 = torch.reshape(q_result_3, [batch_size, -1, 1])

    q_result_t = torch.concat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)   # [B, N, 4]

    q_result = torch.permute(q_result_t, (0, 2, 1)).contiguous()  # [B, 4, N]
    return q_result


def mul_q_point(q, points, scalar_last: bool = False):
    """
    multiplication between Quaternion and points\n
    Inputs:
        * `q`:        [B, 4, 1] or [B, 4]
        * `points`:   [B, 4, N]\n
    Return:
        * `q_result`: [B, 4, N]
    """

    batch_size = points.size(0)
    q = torch.reshape(q, [batch_size, 4, 1])

    q_t = torch.permute(q, (0, 2, 1)).contiguous()  # [B, 1, 4]
    points_t = torch.permute(points, (0, 2, 1)).contiguous()  # [B, N, 4]

    q_result_0 = torch.multiply(q_t[ :, :, 0], points_t[ :, :, 0])-torch.multiply(q_t[ :, :, 1], points_t[ :, :, 1])-torch.multiply(q_t[ :, :, 2], points_t[ :, :, 2])-torch.multiply(q_t[ :, :, 3], points_t[ :, :, 3])
    q_result_0 = torch.reshape(q_result_0, [batch_size, -1, 1])
    
    q_result_1 = torch.multiply(q_t[ :, :, 0], points_t[ :, :, 1])+torch.multiply(q_t[ :, :, 1], points_t[ :, :, 0])+torch.multiply(q_t[ :, :, 2], points_t[ :, :, 3])-torch.multiply(q_t[ :, :, 3], points_t[ :, :, 2])
    q_result_1 = torch.reshape(q_result_1, [batch_size, -1, 1])

    q_result_2 = torch.multiply(q_t[ :, :, 0], points_t[ :, :, 2])-torch.multiply(q_t[ :, :, 1], points_t[ :, :, 3])+torch.multiply(q_t[ :, :, 2], points_t[ :, :, 0])+torch.multiply(q_t[ :, :, 3], points_t[ :, :, 1])
    q_result_2 = torch.reshape(q_result_2, [batch_size, -1, 1])

    
    q_result_3 = torch.multiply(q_t[ :, :, 0], points_t[ :, :, 3])+torch.multiply(q_t[ :, :, 1], points_t[ :, :, 2])-torch.multiply(q_t[ :, :, 2], points_t[ :, :, 1])+torch.multiply(q_t[ :, :, 3], points_t[ :, :, 0])
    q_result_3 = torch.reshape(q_result_3, [batch_size, -1, 1])

    q_result_t = torch.concat([q_result_0, q_result_1, q_result_2, q_result_3], dim=-1)  # [B, N, 4]

    q_result = torch.permute(q_result_t, (0, 2, 1)).contiguous()  # [B, 4, N]
    return q_result
