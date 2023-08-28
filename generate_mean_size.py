

import numpy as np
import open3d
import os

from scipy.spatial.transform import Rotation


datapath = '/run/user/1000/gvfs/sftp:host=10.186.4.18/mnt/isilon/melamine/KITTI-360/heavy_data'  # /run/user/1000/gvfs/sftp:host=10.186.4.18
point_cloud_dir = os.path.join(datapath, 'data_3d_objects')

coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)


rotations = {
    'rider': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    'motorcycle': [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    'bus': [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    'van': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'car': [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    'bicycle': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'person': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    'truck': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
}

str_rotations = ''

file_list = os.listdir(point_cloud_dir)
list_pcd = [coordinate_frame]
nb_obj = 0
for file_name in file_list:
    file_path = os.path.join(point_cloud_dir, file_name)
    extension = os.path.splitext(file_name)[-1]
    if 'obj' not in extension:
        continue

    mesh = open3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)

    obj = os.path.splitext(file_name)[0]
    obj = obj.split('_')[0]
    rot = rotations[obj]

    str_rotations += f'{obj}\n{np.array2string(np.array(rot).reshape(9).astype(int))}\n'

    mesh = mesh.rotate(np.array(rot))
    pcd = pcd.rotate(np.array(rot))

    #pcd = pcd.translate(np.array([0., 0., 5. * nb_obj]))
    nb_obj += 1
 
    list_pcd.append(pcd)

    save_path = os.path.join(point_cloud_dir, f'{obj}.npy')
    np.save(save_path, np.asarray(pcd.points).reshape(-1))

    
    #open3d.io.write_point_cloud(save_path, pcd)

    print(save_path)

    mesh_save_path = os.path.join(point_cloud_dir, f'{obj}.obj')
    #open3d.io.write_triangle_mesh(mesh_save_path, mesh)

    print(mesh_save_path)
    print('')

#rotation_file_path = os.path.join(point_cloud_dir, 'rotations.txt')
#with open(rotation_file_path, 'w') as f:
#    f.write(str_rotations)
#    f.close()

#print('rotations are written in', rotation_file_path)

open3d.visualization.draw_geometries(list_pcd)

