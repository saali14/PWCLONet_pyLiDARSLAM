
import numpy as np
import matplotlib.cm
import random

import open3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import json

import os, sys

# os.environ["RELIDAR_SLAM_ABS_PATH"] = "/run/user/1000/gvfs/sftp:host=10.186.50.16/home/melamine/workspace/relidar-slam"
# os.environ["KITTI360_DATASET"] = "/run/user/1000/gvfs/sftp:host=10.186.50.16/mnt/isilon/melamine/KITTI-360/heavy_data"

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from kitti360Scripts.kitti360scripts.helpers.annotation import Annotation3D, global2local, Annotation3DPly, local2global
from kitti360Scripts.kitti360scripts.helpers.common     import KITTI360_IO, KITTI360_TRANSFORMATIONS, KITTI360_INFOS
from kitti360Scripts.kitti360scripts.helpers.labels     import name2label, id2label, kittiId2label, category2labels
from kitti360Scripts.kitti360scripts.helpers.labels     import labels as LABELS
from kitti360Scripts.kitti360scripts.helpers.ply        import read_ply

from kitti360Scripts.kitti360scripts.viewer.kitti360Viewer3DRaw import Kitti360Viewer3DRaw

# colormap for instances
cmap = matplotlib.colormaps.get_cmap('Set1')
cmap_length = 9 
# colormap for confidence
cmap_conf = matplotlib.colormaps.get_cmap('plasma')

lines_colors = np.tile([1., 0., 0.], (12, 1))
lines_colors[:4, :] = np.array([0., 1., 0.]) # FRONT FACE -- GREEN
lines_colors[4:8, :] = np.array([0., 0., 1.]) # BACK FACE -- BLUE


def read_xml_file(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    super_data = {}
    super_data['scene'] = int(root.find('scene').text)
    super_data['sequence'] = int(root.find('sequence').text)
    super_data['frame_id'] = int(root.find('frame_id').text)
    super_data['window_start'] = int(root.find('window_start').text)
    super_data['window_end'] = int(root.find('window_end').text)

    ego = root.find('ego')
    ego_bbox = extract_bbox_params(ego)

    groups = root.find('groups')
    groups_dict = {}
    objects_dict = {'ego': ego_bbox}
    for group in groups:
        groups_dict[group.tag] = extract_group_params(group)

        objects = group.find('objects')
        for obj in objects:
            objects_dict[obj.tag] = extract_bbox_params(obj)
            objects_dict[obj.tag]['group_id'] = group.tag

    return super_data, groups_dict, objects_dict


def extract_bbox_params(obj: ET.Element):
    bbox = dict()
    bbox['index'] = int(obj.find('index').text)
    bbox['seq_index'] = int(obj.find('seq_index').text)
    bbox['semanticId'] = int(obj.find('semanticId').text)
    bbox['instanceId'] = int(obj.find('instanceId').text)
    bbox['categoryId'] = int(obj.find('categoryId').text)
    bbox['label'] = obj.find('label').text
    bbox['isDynamic'] = bool(obj.find('isDynamic').text)
    bbox['distance'] = float(obj.find('distance').text)
    bbox['speed'] = float(obj.find('speed').text)

    bbox['center'] = np.fromstring(obj.find('center').text, dtype=float, sep=',').reshape(3)
    bbox['corners'] = np.fromstring(obj.find('corners').text, dtype=float, sep=',').reshape(8, 3)
    bbox['size'] = np.fromstring(obj.find('size').text, dtype=float, sep=',').reshape(3)

    bbox['heading'] = float(obj.find('heading').text)
    bbox['heading_class'] = int(obj.find('heading_class').text)       # ///////!!!!!!!\\\\\\\\
    bbox['heading_residual'] = float(obj.find('heading_residual').text)
    bbox['relative_angle'] = float(obj.find('relative_angle').text)

    bbox['description'] = obj.find('description').text

    bbox['bbox'] = compute_bbox(bbox)

    return bbox


def compute_bbox(bbox_dict):
    corners = bbox_dict['corners']
    lines = KITTI360_INFOS.CAR_LINES
    bbox = open3d.geometry.LineSet(
        points = open3d.utility.Vector3dVector(corners),
        lines = open3d.utility.Vector2iVector(lines)
    )
    bbox.colors = open3d.utility.Vector3dVector(lines_colors)

    return bbox


def extract_group_params(group: ET.Element):
    group_dict = dict()
    group_dict['front'] = group.find('front').text
    group_dict['side'] = group.find('side').text
    group_dict['isDynamic'] = bool(group.find('isDynamic').text)
    group_dict['categoryId'] = int(group.find('categoryId').text)
    group_dict['label'] = group.find('label').text
    group_dict['nb_objects'] = int(group.find('nb_objects').text)
    group_dict['indexes'] = np.fromstring(group.find('indexes').text, dtype=int, sep=',').reshape(-1)

    group_dict['description'] = group.find('description').text

    return group_dict


def transformPoints(points: np.ndarray, pose: np.ndarray):
    new_points = np.matmul(np.linalg.inv(pose[:3,:3]), (points - pose[:3,3].transpose()).transpose()).transpose()
    return new_points


def cropPcd(pcd: open3d.geometry.PointCloud, radius: float):
    pcdPoints = np.asarray(pcd.points)
    pcdColors = np.asarray(pcd.colors)

    distances = np.linalg.norm(pcdPoints[:, :2], axis=1)
    idx_points_in_range = np.where(distances <= radius)[0]
    points_in_range = pcdPoints[idx_points_in_range,:]
    new_colors = pcdColors[idx_points_in_range,:]

    pcd.points = open3d.utility.Vector3dVector(points_in_range)
    pcd.colors = open3d.utility.Vector3dVector(new_colors)

    return pcd


def compute_instance_bboxes(objects_dict):
    max_nb_objs = 50
    instance_bboxes = {
        'bbox': np.zeros((max_nb_objs, 6)),
        'semanticId': np.zeros((max_nb_objs), dtype=int),
        'instanceId': np.zeros((max_nb_objs), dtype=int),
        'categoryId': np.zeros((max_nb_objs), dtype=int),
        'heading': np.zeros((max_nb_objs), dtype=float),
        'isDynamic': np.zeros((max_nb_objs), dtype=bool)
    }

    nb_bboxes = 0
    for i, obj_id in enumerate(objects_dict):
        if i >= max_nb_objs:
            break
        instance_bboxes['bbox'][i,:3]    = objects_dict[obj_id]['center']
        instance_bboxes['bbox'][i,3:6]   = objects_dict[obj_id]['size']
        instance_bboxes['semanticId'][i] = objects_dict[obj_id]['semanticId']
        instance_bboxes['instanceId'][i] = objects_dict[obj_id]['instanceId']
        instance_bboxes['categoryId'][i] = objects_dict[obj_id]['categoryId']
        instance_bboxes['heading'][i]    = objects_dict[obj_id]['heading']
        instance_bboxes['isDynamic'][i]  = objects_dict[obj_id]['isDynamic']
        
        nb_bboxes += 1

    return instance_bboxes, nb_bboxes


def get_max_distance(objects_dict):
    max_dist = 0.
    for obj_id in objects_dict.keys():
        obj_corners = objects_dict[obj_id]['corners']
        obj_corners_dist = np.linalg.norm(obj_corners[:,:2], axis=1)
        obj_max_dist = np.max(obj_corners_dist)
        max_dist = max(max_dist, obj_max_dist)
        #obj_dist = objects_dict[obj_id]['distance']
        #max_dist = max(max_dist, obj_dist)
    return max_dist


def adjustPcd(pcd: open3d.geometry.PointCloud, frame: int, objects_dict: dict, pose):
    pcdPoints = np.asarray(pcd.points)
    new_points = transformPoints(pcdPoints, pose)
    pcd.points = open3d.utility.Vector3dVector(new_points)

    instance_bboxes, nb_bboxes = compute_instance_bboxes(objects_dict)
    pcd = completePcd(pcd, instance_bboxes)

    max_dist = get_max_distance(objects_dict)
    return cropPcd(pcd, max_dist)


def completePcd(pcd: open3d.geometry.PointCloud, instance_bboxes):
    unique_semanticIds = np.unique(instance_bboxes['semanticId'])
    dynamic_objs = instance_bboxes['isDynamic']

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.points)

    if np.any(dynamic_objs):
        for semanticId in unique_semanticIds:
            # if we have processed all the dynamic objects
            if not np.any(dynamic_objs):
                break
            inds = (instance_bboxes['semanticId'] == semanticId)
            final_inds = np.logical_and(dynamic_objs, inds)
            if np.any(final_inds):
                label_name = id2label[semanticId].name
                point_cloud_path = os.path.join(kitti360Path, 'data_3d_objects', f'{label_name}.obj')
                if not os.path.exists(point_cloud_path):
                    dynamic_objs[final_inds] = False
                    continue
                mesh = open3d.io.read_triangle_mesh(point_cloud_path)
                pcd = mesh.sample_points_uniformly(number_of_points=1000)

                indexes = np.where(final_inds)[0]
                bboxes = instance_bboxes['bbox'][indexes,:]                     # nb_bboxes, 6
                headings = instance_bboxes['heading'][indexes]                  # nb_bboxes
                bboxes_instances_ids = instance_bboxes['instanceId'][indexes]   # nb_bboxes

                pcd_points = np.asarray(pcd.points)
                nb_bboxes = bboxes.shape[0]
                nb_points = pcd_points.shape[0]

                new_points = np.tile(pcd_points, (nb_bboxes,1))                 # nb_bboxes * nb_points, 3

                globalIds = local2global(np.tile(semanticId, (nb_bboxes)), bboxes_instances_ids)
                new_colors = assignColor(globalIds)    # nb_bboxes
                new_colors = np.tile(new_colors, (1,1,nb_points)).reshape((new_points.shape[0],3))     # nb_bboxes * nb_points, 3

                #new_semanticIds = np.tile(semanticId, (new_points.shape[0]))    # nb_bboxes * nb_points,
                #new_intanceIds = np.tile(bboxes_instances_ids.reshape(nb_bboxes,1), (1,nb_points)).reshape((new_points.shape[0]))   # nb_bboxes * nb_points,

                length = np.max(pcd_points[:,0], axis=0) - np.min(pcd_points[:,0], axis=0)
                width = np.max(pcd_points[:,1], axis=0) - np.min(pcd_points[:,1], axis=0)
                height = np.max(pcd_points[:,2], axis=0) - np.min(pcd_points[:,2], axis=0)

                scale_length = (bboxes[:,-3] / length).reshape(nb_bboxes,1)    # nb_bboxes, 1
                scale_width  = (bboxes[:,-2] / width).reshape(nb_bboxes,1)     # nb_bboxes, 1
                scale_height = (bboxes[:,-1] / height).reshape(nb_bboxes,1)    # nb_bboxes, 1
                scale = np.mean(np.concatenate((scale_length, scale_width, scale_height), axis=1), axis=1, keepdims=True)  # nb_bboxes, 1
                #scale = np.tile(scale, (1, nb_points)).reshape(new_points.shape[0]) # nb_bboxes * nb_points

                new_points = new_points.reshape((nb_bboxes, nb_points, 3)) # nb_bboxes, N, 3

                new_points = np.multiply(new_points, scale.reshape((nb_bboxes, 1, 1))) # nb_bboxes, nb_points, 3

                # so that the objects be on the floor
                min_heights = np.min(new_points[:,:,2], axis=1) # nb_bboxes, 
                #pos_height_inds = np.where(min_heights > 0)
                #factors = np.ones((nb_bboxes))   # points with negative heights should be added
                #factors[pos_height_inds] = -1.      # points with positive heights should be substracted
                new_points[:,:,2] = new_points[:,:,2] - min_heights.reshape((nb_bboxes, 1))

                transformations = np.tile(np.eye(4), (nb_bboxes, 1, 1))
                transformations[:,:3,3] = bboxes[:,:3]

                r = Rotation.from_euler('z', headings, degrees=False)
                rotations = r.as_matrix()

                transformations[:,:3,:3] = rotations

                add_T = np.tile(np.ones((nb_points, 1)), (nb_bboxes, 1, 1))
                points_t = np.concatenate((new_points, add_T), axis = -1)
                points_t = np.matmul(transformations, np.transpose(points_t, (0, 2, 1)))
                new_points = np.transpose(points_t, (0, 2, 1))[:, :, :3]

                new_points = new_points.reshape((nb_bboxes * nb_points, 3))

                points = np.concatenate((points, new_points), axis=0)
                colors = np.concatenate((colors, new_colors), axis=0)
                #semanticIds = np.concatenate((semanticIds, new_semanticIds), axis=0)
                #intanceIds = np.concatenate((intanceIds, new_intanceIds), axis=0)

                dynamic_objs[final_inds] = False

    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd


def loadColors(labels, colorType='semantic'):

    if (colorType == 'reflectance'):
        reflectance = labels
        colors = assignColorReflectance(reflectance)      
    elif (colorType == 'semantic'):
        semanticIds = labels[:,3]
        colors = assignColorSemantic(semanticIds)
    elif (colorType == 'instance'): 
        # the problem here is that when writing the labels we specified np.int16 as their type
        # however `instanceID` takes large values that exceeds 16 bits 
        globalIds = labels[:,4]
        colors = assignColor(globalIds, colorType)
    elif colorType == 'confidence':
        confidence = labels[:,-1]
        colors = assignColorConfidence(confidence)
    elif (colorType == 'rgb') or (colorType == 'bbox'):
        colors = labels[:,:3] # rgb color
        colors = colors.astype(np.float64)/255.
    else:
        raise RuntimeError(f'Unknown `colorType`: {colorType}')

    return colors


def assignColorSemantic(semanticIds):
    if not isinstance(semanticIds, (np.ndarray, np.generic)):
        semanticIds = np.array(semanticIds)[None]
    color = np.zeros((semanticIds.size, 3))
    for uid in np.unique(semanticIds):
        color[semanticIds==uid] = id2label[uid].color
    color = color.astype(np.float64)/255.0
    return color


def assignColorReflectance(reflectance):
    color = cmap_conf(reflectance)[:,:3]
    return color


def getColor(idx):
    if idx == 0:
        return np.array([0,0,0])
    return np.asarray(cmap(idx % cmap_length)[:3])*255.


def assignColor(globalIds, gtType='semantic'):
    if not isinstance(globalIds, (np.ndarray, np.generic)):
        globalIds = np.array(globalIds)[None] # -- ME -- add a new axis
    color = np.zeros((globalIds.size, 3)) # -- ME -- (1, 4, 5) -> size = 4
    for uid in np.unique(globalIds):
        semanticId, instanceId = global2local(uid)
        if gtType == 'semantic':
            color[globalIds==uid] = id2label[semanticId].color
        elif instanceId > 0:
            color[globalIds==uid] = getColor(instanceId)
        else:
            color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
    color = color.astype(np.float64)/255.0
    return color


def assignColorConfidence(confidence):
    color = cmap_conf(confidence)[:,:3]
    return color


def assignColorDynamic(timestamps):
    color = np.zeros((timestamps.size, 3))
    for uid in np.unique(timestamps):
        color[timestamps==uid] = getColor(uid)
    return color


def loadWindow(pcdFile, mode):
    window = pcdFile.split(os.sep)[-2] # static or dynamic
    
    print ('Loading %s ' % pcdFile)

    #pcd = open3d.io.read_point_cloud(pcdFile)
    data = read_ply(pcdFile)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    color = np.vstack((data['red'], data['green'], data['blue'])).T

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(color.astype(np.float64)/255.)        
    
    # assign color
    if mode == 'semantic' or mode == 'instance':
        globalIds = data['instance']
        ptsColor = assignColor(globalIds, mode)
        pcd.colors = open3d.utility.Vector3dVector(ptsColor)

    elif mode == 'bbox':
        ptsColor = np.asarray(pcd.colors)
        pcd.colors = open3d.utility.Vector3dVector(ptsColor)

    elif mode == 'confidence':
        # replaced next line
        #confidence = data[:,-1]
        confidence = data['confidence']
        ptsColor = assignColorConfidence(confidence)
        pcd.colors = open3d.utility.Vector3dVector(ptsColor)

    elif mode != 'rgb':
        raise ValueError("Color type can only be 'rgb', 'bbox', 'semantic', 'instance'!")

    return pcd


def on_key(e):
    if e.key == gui.KeyName.SPACE:
        if e.type == gui.KeyEvent.UP:
            pass
        else:
            pass
        return gui.Widget.EventCallbackResult.HANDLED
    if e.key == gui.KeyName.W:  # eats W, which is forward in fly mode
        return gui.Widget.EventCallbackResult.CONSUMED
    return gui.Widget.EventCallbackResult.IGNORED


def visualize_map(data_path, seq, mode, showStatic: bool = True):
    sequence = KITTI360_IO.drive_foldername(seq)

    poses_dir = os.path.join(kitti360Path, 'data_poses')
    poses_file = os.path.join(poses_dir, sequence, 'poses.txt')
    poses = KITTI360_IO.loadPoses(poses_file, sep=' ', velo_to_world=True)

    if showStatic:
        label3DPcdPath  = os.path.join(data_path, 'data_maps')
        
        pcdFileDir = 'static' # 'dynamic' if not cap_visualizer.showStatic else 'static'
        ply_dir_path = os.path.join(label3DPcdPath, 'train', sequence, pcdFileDir)
        ply_files = os.listdir(ply_dir_path)
        pcdFileName = random.choice(ply_files)
        pcdFilePath = os.path.join(ply_dir_path, pcdFileName)
        points, colors, semanticIds, instanceIds = KITTI360_IO.loadWindowNpy(pcdFilePath)
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors) 
        
    else:
        label3DPcdPath = os.path.join(kitti360Path, 'data_3d_semantics')

        pcdFileDir = 'static' # 'dynamic' if not cap_visualizer.showStatic else 'static'
        ply_dir_path = os.path.join(label3DPcdPath, 'train', sequence, pcdFileDir)
        ply_files = os.listdir(ply_dir_path)
        pcdFileName = random.choice(ply_files)
        pcdFilePath = os.path.join(ply_dir_path, pcdFileName)
        pcd = loadWindow(pcdFilePath, mode)

    car_pcd_path = os.path.join(kitti360Path, 'data_3d_objects', f'car.npy')
    ego_vehicle_points = np.load(car_pcd_path)
    ego_vehicle_points = ego_vehicle_points.reshape((-1, 3))

    frames = sorted(list(poses.keys()))
    ego_vehicle_point_cloud = np.zeros((len(frames), ego_vehicle_points.shape[0], 3))

    pcdFileNameBase = os.path.splitext(pcdFileName)[0]
    start_window = int(pcdFileNameBase.split('_')[0])
    end_window = int(pcdFileNameBase.split('_')[-1])

    captions_data_path = os.path.join(kitti360Path, 'data_captions')
    """
    meta_data_path = os.path.join(captions_data_path, 'meta_data.txt')
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    f.close()
    scene_ids = meta_data[str(seq)][f'{start_window}_{end_window}']
    """

    nb_frames = 0
    for frame in frames:
        pose = poses[frame]

        if (frame < start_window) or (frame > end_window):
            continue

        points = np.copy(ego_vehicle_points)
        add_T = np.ones((points.shape[0], 1))
        points = np.concatenate((points, add_T), axis = -1)
        points = np.matmul(pose, np.transpose(points, (1, 0)))
        points = np.transpose(points, (1, 0))[:, :3]

        ego_vehicle_point_cloud[nb_frames,:,:] = points
        nb_frames += 1

    ego_vehicle_point_cloud = ego_vehicle_point_cloud[:nb_frames,:,:]

    ego_vehicle_colors = np.tile(np.array([1., 0., 0.]), (nb_frames * ego_vehicle_points.shape[0], 1))

    ego_vehicle_pcd = open3d.geometry.PointCloud()
    ego_vehicle_pcd.points = open3d.utility.Vector3dVector(ego_vehicle_point_cloud.reshape((-1, 3)))
    ego_vehicle_pcd.colors = open3d.utility.Vector3dVector(ego_vehicle_colors)

    open3d.visualization.draw_geometries([pcd, ego_vehicle_pcd])

    ###########################
    # BBOXES
    ###########################

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Caption Visualizer", 1024, 768)
    widget3d = gui.SceneWidget()
    window.add_child(widget3d)
    widget3d.scene = rendering.Open3DScene(window.renderer)

    widget3d.scene.show_axes(False)
    widget3d.scene.show_skybox(False)

    #widget3d.set_on_key(on_key)

    # --------------------

    bboxes = []
    ann_ids = []
    nb_scenes = 0
    
    motion_name = 'static' if showStatic else 'dynamic'
    window_directory = os.path.join(captions_data_path, motion_name, sequence, pcdFileNameBase)
    scene_files = os.listdir(window_directory)
    
    print('Meta Data:\n')
    print(pcdFileNameBase)
    print('')
    
    for scene_file in scene_files:
        xml_file_path = os.path.join(window_directory, scene_file)
        # xml_file_path = os.path.join(window_directory, f'{str(scene_id).zfill(6)}.xml')
        super_data, groups_dict, objects_dict = read_xml_file(xml_file_path)

        frame = super_data['frame_id']
        if frame not in frames:
            continue
        pose = poses[frame]

        color = np.asarray(cmap(nb_scenes % cmap_length)[:3])

        # this is for captioning the groups
        # it doesn't work
        """
        for group_id in groups_dict:
            obj_ids = groups_dict[group_id]['indexes']
            corners = np.zeros((len(obj_ids), 8, 3))
            for i, obj_id in enumerate(obj_ids):
                corners[i,:,:] = objects_dict[f'object_{obj_id}']['corners']

            corners = corners.reshape((-1, 3))
            ids = np.argsort(np.linalg.norm(corners, axis=1))
            group_corners = corners[ids[:8]]

            lines = KITTI360_INFOS.CAR_LINES
            bbox = open3d.geometry.LineSet(
                points = open3d.utility.Vector3dVector(group_corners),
                lines = open3d.utility.Vector2iVector(lines)
            )
            bbox.colors = open3d.utility.Vector3dVector(np.tile(color, (np.asarray(lines).shape[0], 1)))

            bbox = bbox.transform(pose)

            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            widget3d.scene.add_geometry(group_id, bbox, mat)

            geometry_label = open3d.geometry.PointCloud()
            geometry_label_point = bbox.get_center()
            geometry_label.points = open3d.utility.Vector3dVector(geometry_label_point.reshape(-1, 3))
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 5 * window.scaling
            mat.base_color = (0, 0, 0, 1)
            widget3d.scene.add_geometry(group_id, geometry_label, mat)

            description = groups_dict[group_id]['description']
            #tokens = description.split(' ')
            #nb_tokens = len(tokens)
            #description = ' '.join(tokens[:nb_tokens//2]) + '\n' + ' '.join(tokens[nb_tokens//2:])
            label = widget3d.add_3d_label(geometry_label_point, description)
            label.color = gui.Color(color[0], color[1], color[2])
            label.scale = 1.3
        """

        for obj_id in objects_dict:
            #if 'ego' in obj_id:
            #    continue

            ann_id = objects_dict[obj_id]['index']
            if ann_id in ann_ids:
                continue
            ann_ids.append(ann_id)

            obj_bbox = open3d.geometry.LineSet(objects_dict[obj_id]['bbox'])
            obj_bbox = obj_bbox.transform(pose)

            #if 'ego' in obj_id:
            #    obj_bbox.colors = open3d.utility.Vector3dVector(np.tile(color, (np.asarray(obj_bbox.lines).shape[0], 1)))

            bboxes.append(obj_bbox)

            ############################

            if 'ego' in obj_id:
                continue

            object_name = f'{objects_dict[obj_id]["label"]}_{ann_id}'

            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            widget3d.scene.add_geometry(object_name, obj_bbox, mat)

            geometry_label = open3d.geometry.PointCloud()
            geometry_label_point = obj_bbox.get_center()
            geometry_label.points = open3d.utility.Vector3dVector(geometry_label_point.reshape(-1, 3))
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 5 * window.scaling
            mat.base_color = (0, 0, 0, 1)
            widget3d.scene.add_geometry(object_name, geometry_label, mat)

            description = objects_dict[obj_id]['description']
            
            print('')
            print(f'[{obj_id}] {description}')
            print('')
            
            tokens = description.split(' ')
            nb_tokens = len(tokens)
            description = ' '.join(tokens[:nb_tokens//2]) + '\n' + ' '.join(tokens[nb_tokens//2:])
            label = widget3d.add_3d_label(geometry_label_point, description)
            label.color = gui.Color(color[0], color[1], color[2])
            label.scale = 1.3
        
        coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        coordinate_frame = coordinate_frame.transform(pose)
        
        scene_bboxes = []
        for obj_id in objects_dict.keys():

            obj_bbox = open3d.geometry.LineSet(objects_dict[obj_id]['bbox'])
            obj_bbox = obj_bbox.transform(pose)

            scene_bboxes.append(obj_bbox)
            
        print(f'Scene {scene_file} contains {len(scene_bboxes)} objects')
        print('\n#########################################################\n')
            
        scene_bboxes.append(pcd)
        scene_bboxes.append(coordinate_frame)               
            
        open3d.visualization.draw_geometries(scene_bboxes)
        
        nb_scenes += 1

    geometries = bboxes + [pcd, ego_vehicle_pcd]

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultLit"
    widget3d.scene.add_geometry('point_cloud', pcd, mat_pcd)

    mat_ego = rendering.MaterialRecord()
    mat_ego.shader = "defaultLit"
    widget3d.scene.add_geometry('ego_vehicle', ego_vehicle_pcd, mat_ego)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60, bounds, bounds.get_center())

    app.run()

    input("Press Enter to close the visualization")

    app.quit()


if __name__ == '__main__':
    if 'KITTI360_DATASET' in os.environ:
        kitti360Path = os.environ['KITTI360_DATASET']
    else:
        kitti360Path = os.path.join(os.path.dirname(
                            os.path.realpath(__file__)), '..', '..')
    
    seq = 0
    mode = 'semantic'

    visualize_map(kitti360Path, seq, mode)