
import numpy as np
import time
import matplotlib.cm
from typing import List, Dict
import argparse
import math
from scipy.spatial.transform import Rotation

import open3d

import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from kitti360Scripts.kitti360scripts.helpers.annotation import Annotation3D, global2local
from kitti360Scripts.kitti360scripts.helpers.common     import KITTI360_IO, KITTI360_INFOS
from kitti360Scripts.kitti360scripts.helpers.labels     import id2label, category2labels
from kitti360Scripts.kitti360scripts.helpers.labels     import labels as LABELS
from kitti360Scripts.kitti360scripts.helpers.ply        import read_ply

import shutil


class KITTI360Maps:

    SIDE_TRESHOLD = KITTI360_INFOS.CAR_WIDTH
    FRONT_TRESHOLD = KITTI360_INFOS.CAR_LENGTH
    
    # Constructor
    def __init__(self, config: Dict):

        self.downSampleEvery   = -1

        self.seq            = config['sequence']
        self.showStatic     = config['showStatic']
        self.selectedlabels = config['labels']
        self.withGround     = config['withGround']
        self.num_points     = config['num_points']

        self.num_points_objs = self.num_points//2
        self.num_points_map = self.num_points - self.num_points_objs

        # show visible point clouds only
        self.showVisibleOnly   = False

        # colormap for instances
        self.cmap = matplotlib.colormaps.get_cmap('Set1')
        self.cmap_length = 9 
        # colormap for confidence
        self.cmap_conf = matplotlib.colormaps.get_cmap('plasma')

        if 'KITTI360_DATASET' in os.environ:
            self.kitti360Path = os.environ['KITTI360_DATASET']
        else:
            self.kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

        sequence = KITTI360_IO.drive_foldername(self.seq)
        self.sequence = sequence

        self.label3DBboxPath = os.path.join(self.kitti360Path, 'data_3d_bboxes')
        self.annotation3D = Annotation3D(self.label3DBboxPath, sequence, with_ground=self.withGround, verbose=0)

        self.plyDirPath = os.path.join(self.kitti360Path, 'data_3d_semantics', 'train', self.sequence, 'static')
        self.listPlyFiles = os.listdir(self.plyDirPath)

        self.save_dir = config['output_dir']
        motion_name = 'static' if self.showStatic else 'dynamic'
        if self.save_dir is None:
            self.save_dir = os.path.join(self.kitti360Path, 'data_maps', 'train', self.sequence, motion_name)

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        
        os.makedirs(self.save_dir)

        self.bboxes = []
        self.bboxes_dict = []
        self.bboxes_window = []
        self.windows_unique = []

        self.labels = []

        self.globalIds = []

        self.front_vertices_idx = [0, 1, 3, 2]
        self.back_vertices_idx = [4, 5, 7, 6]
        self.left_vertices_idx = [0, 1, 4, 5]
        self.right_vertices_idx = [2, 3, 6, 7]
        self.up_vertices_idx = [0, 2, 5, 7]
        self.down_vertices_idx = [1, 3, 4, 6]

        self.lines_colors = np.tile([1., 0., 0.], (12, 1))
        self.lines_colors[4:8, :] = np.array([0., 1., 0.]) # closest face
        self.lines_colors[8:, :] = np.array([0., 0., 1.]) # farthest face

        self.init()
        

    def init(self):
        self.loadBoundingBoxes()
        if not len(self.bboxes):
            raise RuntimeError('No bounding boxes found! Please set KITTI360_DATASET in your environment path')


    def run(self):
        print(f'>> Map Point Cloud dataset generation for sequence {self.seq} ...')
        print(f'>> Processing {len(self.listPlyFiles)} windows ...')
        print(f'{len(self.bboxes)} objects were detected')

        for plyFile in self.listPlyFiles:

            window_start, window_end = KITTI360_IO.window_from_file_name(plyFile)
            window = [window_start, window_end]

            pcdFilePath = os.path.join(self.plyDirPath, plyFile)

            points, colors, semanticIds, instanceIds = self.loadWindow(pcdFilePath)

            indices = self.cropPcd(points, self.num_points_map)
            points = points[indices,:]
            colors = colors[indices,:]
            semanticIds = semanticIds[indices]
            instanceIds = instanceIds[indices]

            bboxes = self.get_bboxes(window, points)
            instance_bboxes, nb_bboxes = self.compute_instance_bboxes(bboxes)

            if nb_bboxes == 0:
                continue

            new_points, new_colors, new_semanticIds, new_instanceIds = self.completePcd(instance_bboxes, nb_bboxes)

            points = np.concatenate((points, new_points), axis=0)
            colors = np.concatenate((colors, new_colors), axis=0)
            semanticIds = np.concatenate((semanticIds, new_semanticIds), axis=0)
            instanceIds = np.concatenate((instanceIds, new_instanceIds), axis=0)
            
            indices = self.cropPcd(points, self.num_points)
            points = points[indices,:]
            colors = colors[indices,:]
            semanticIds = semanticIds[indices]
            instanceIds = instanceIds[indices]

            """
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points[:,:3])
            pcd.colors = open3d.utility.Vector3dVector(colors[:,:3])

            open3d.visualization.draw_geometries([pcd])
            exit()
            """
            
            save_array = np.concatenate((points.reshape((-1,3)), colors.reshape((-1,3)), semanticIds.reshape((-1,1)), instanceIds.reshape((-1,1))), axis=1)
            save_file = f'{os.path.splitext(plyFile)[0]}.npy'
            save_path = os.path.join(self.save_dir, save_file)
            with open(save_path, 'wb') as f:
                np.save(f, save_array)
            
        print(f'>> Maps were generated successfully.\nCheck the following directory {self.save_dir}\n')


    def loadWindow(self, pcdFile):

        data = read_ply(pcdFile)

        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.vstack((data['red'], data['green'], data['blue'])).T
        colors = colors / 255.

        # assign color
        globalIds = data['instance']
        semanticIds = np.zeros(points.shape[0])
        instanceIds = np.zeros(points.shape[0])

        """
        for uid in np.unique(globalIds):
            semanticId, instanceId = global2local(uid)
            semanticIds[globalIds == uid] = semanticId
            instanceIds[globalIds == uid] = instanceId
        """

        semanticIds, instanceIds = global2local(globalIds)

        if self.showVisibleOnly:
            isVisible = data['visible']
            inds = np.where(isVisible)[0]
            points = points[inds,:]
            colors = colors[inds,:]
            semanticIds = semanticIds[inds]
            instanceIds = instanceIds[inds]

        return points, colors, semanticIds, instanceIds


    def reset_containers(self):
        self.bboxes = []
        self.bboxes_dict = []
        self.bboxes_window = []
        self.windows_unique = []
        self.labels = []
        self.globalIds = []


    def loadBoundingBoxes(self):

        self.reset_containers()

        for i, (globalId, v) in enumerate(self.annotation3D.objects.items()):
            # skip dynamic objects

            semanticId, instanceId = global2local(globalId)

            if (self.selectedlabels is not None) and (len(self.selectedlabels) > 0) and (id2label[semanticId].name not in self.selectedlabels):
                continue
            
            if len(v) > 1:
                if self.showStatic:
                    continue
                first_key = list(v.keys())[0]
                self.windows_unique.append([v[first_key].start_frame, v[first_key].end_frame])
                keys = list(v.keys())
                key = np.random.choice(keys)
                obj = v[key]
            else:
                # for static object, they only have one bounding box that has a timestamp equal to -1
                self.windows_unique.append([v[-1].start_frame, v[-1].end_frame])
                obj = v[-1]
                
            lines = np.array(obj.lines)
            vertices = np.array(obj.vertices)
            faces = np.array(obj.faces)

            lines_colors = self.lines_colors
            if self.withGround:
                lines = self.getLinesFromTriangles(faces)
                _, _, front_faces_idx = self.getFace(vertices, lines, 'front')
                _, _, back_faces_idx = self.getFace(vertices, lines, 'back')
                lines_colors = np.tile([1., 0., 0.], (len(lines), 1))
                lines_colors[front_faces_idx,:] = np.array([0., 1., 0.]) 
                lines_colors[back_faces_idx,:] = np.array([0., 0., 1.]) 

            bbox = open3d.geometry.LineSet(
                points = open3d.utility.Vector3dVector(vertices),
                lines = open3d.utility.Vector2iVector(lines)
            )
            bbox.colors = open3d.utility.Vector3dVector(lines_colors)

            bbox_dict = self.process_bbox(bbox, obj)

            self.bboxes.append(bbox)
            self.bboxes_dict.append(bbox_dict)
            self.bboxes_window.append([obj.start_frame, obj.end_frame])
            self.labels.append(id2label[semanticId].name)
            self.globalIds.append(globalId)                

        self.windows_unique = np.unique(np.array(self.windows_unique), axis=0)


    def get_bboxes(self, window, points):
        """
        bboxes: dict
        """

        max_coord = np.max(points, axis=0)
        min_coord = np.min(points, axis=0)
        
        bboxes = []
        for i in range(len(self.bboxes)):
            #if (self.bboxes_window[i][0] >= window[0]) and (self.bboxes_window[i][1] <= window[1]):
            globalId = self.globalIds[i]
            #obj = self.annotation3D.objects[globalId][-1]
            bbox = self.bboxes[i]
            bbox_dict = self.bboxes_dict[i]

            center = bbox.get_center()
            if (center[0] <= max_coord[0]) and (center[1] <= max_coord[1]) \
                and (center[0] >= min_coord[0]) and (center[1] >= min_coord[1]):
                bboxes.append(bbox_dict)

        return bboxes
    

    def process_bbox(self, bbox, obj):
        bbox_center = bbox.get_center()

        bbox_dict = {
            'index': obj.annotationId,
            'seq_index': obj.seq_index,
            'semanticId': obj.semanticId,
            'instanceId': obj.instanceId,
            'categoryId': id2label[obj.semanticId].categoryId,
            'bbox': bbox,
            'isDynamic': obj.isDynamic
        }

        bbox_dict['label'] = id2label[obj.semanticId].name

        bbox_center, bbox_corners, bbox_heading, bbox_size = self.compute_bbox_params(bbox)
        bbox_dict['center'] = bbox_center
        bbox_dict['corners'] = bbox_corners
        bbox_dict['heading'] = bbox_heading
        bbox_dict['size'] = bbox_size

        return bbox_dict


    def compute_bbox_params(self, bbox):
        """
        bbox_center, bbox_corners, bbox_heading (in radians between 0 and 2*pi), bbox_size
        """

        bbox_center = bbox.get_center()

        if isinstance(bbox, open3d.geometry.LineSet):
            bbox_corners = np.asarray(bbox.points)
            faces = np.asarray(bbox.lines)
        elif isinstance(bbox, open3d.geometry.TriangleMesh):
            bbox_corners = np.asarray(bbox.vertices)
            faces = np.asarray(bbox.triangles)
        else:
            raise RuntimeError('[compute_bbox_params] Bouding Box should be either LineSet or TriangleMesh')
        
        front_corners, front_faces, _ = self.getFace(bbox_corners, faces, 'front')
        right_corners, right_faces, _ = self.getFace(bbox_corners, faces, 'right')
        up_corners, up_faces, _ = self.getFace(bbox_corners, faces, 'up')

        if isinstance(bbox, open3d.geometry.LineSet):            
            front_face_mesh = open3d.geometry.LineSet(
                points = open3d.utility.Vector3dVector(front_corners),
                lines = open3d.utility.Vector2iVector(front_faces)
            )

            right_face_mesh = open3d.geometry.LineSet(
                points = open3d.utility.Vector3dVector(right_corners),
                lines = open3d.utility.Vector2iVector(right_faces)
            )

            up_face_mesh = open3d.geometry.LineSet(
                points = open3d.utility.Vector3dVector(up_corners),
                lines = open3d.utility.Vector2iVector(up_faces)
            )
        else:
            front_face_mesh = open3d.geometry.TriangleMesh(
                vertices = open3d.utility.Vector3dVector(front_corners),
                triangles = open3d.utility.Vector3iVector(front_faces)
            )

            right_face_mesh = open3d.geometry.TriangleMesh(
                vertices = open3d.utility.Vector3dVector(right_corners),
                triangles = open3d.utility.Vector3iVector(right_faces)
            )

            up_face_mesh = open3d.geometry.TriangleMesh(
                vertices = open3d.utility.Vector3dVector(up_corners),
                triangles = open3d.utility.Vector3iVector(up_faces)
            )

        front_face_center = front_face_mesh.get_center()
        right_face_center = right_face_mesh.get_center()
        up_face_center = up_face_mesh.get_center()

        # compute bbox heading in the z-axis from x-axis CCW
        front_relative_pose = (front_face_center[0] - bbox_center[0], front_face_center[1] - bbox_center[1])
        bbox_orientation = math.atan2(front_relative_pose[1], front_relative_pose[0])
        bbox_heading = bbox_orientation % (2*np.pi)

        # compute bbox size
        length = math.dist(front_face_center[:2], bbox_center[:2]) * 2.
        width = math.dist(right_face_center[:2], bbox_center[:2]) * 2.
        height = np.linalg.norm(up_face_center[2] - bbox_center[2]) * 2.

        bbox_size = np.asarray([length, width, height])

        return bbox_center, bbox_corners, bbox_heading, bbox_size
    

    def compute_instance_bboxes(self, objects_dict):
        """
        * `bbox`: center (xyz) and size (lwh)
        * `semanticId`
        * `instanceId`
        * `categoryId`
        * `heading`
        * `isDynamic`
        """

        instance_bboxes = {
            'bbox': [],
            'semanticId': [],
            'instanceId': [],
            'categoryId': [],
            'heading': [],
            'isDynamic': []
        }

        nb_bboxes = 0
        for i, obj_dict in enumerate(objects_dict):
            bbox = np.zeros((1,6))
            bbox[0,:3] = obj_dict['center']
            bbox[0,3:6] = obj_dict['size']
            instance_bboxes['bbox'].append(bbox)
            instance_bboxes['semanticId'].append(obj_dict['semanticId'])
            instance_bboxes['instanceId'].append(obj_dict['instanceId'])
            instance_bboxes['categoryId'].append(obj_dict['categoryId'])
            instance_bboxes['heading'].append(obj_dict['heading'])
            instance_bboxes['isDynamic'].append(obj_dict['isDynamic'])
            
            nb_bboxes += 1

        if nb_bboxes > 0:
            instance_bboxes['bbox'] = np.concatenate(instance_bboxes['bbox'], axis=0)
            instance_bboxes['semanticId'] = np.array(instance_bboxes['semanticId'])
            instance_bboxes['instanceId'] = np.array(instance_bboxes['instanceId'])
            instance_bboxes['categoryId'] = np.array(instance_bboxes['categoryId'])
            instance_bboxes['heading'] = np.array(instance_bboxes['heading'])
            instance_bboxes['isDynamic'] = np.array(instance_bboxes['isDynamic'])

        return instance_bboxes, nb_bboxes


    def completePcd(self, instance_bboxes, nb_bboxes):
        num_points_per_obj = self.num_points_objs // nb_bboxes

        unique_semanticIds = np.unique(instance_bboxes['semanticId'])

        #dynamic_objs = instance_bboxes['isDynamic'].astype(bool)
        # consider all objects not only dynamic
        dynamic_objs = np.ones(instance_bboxes['isDynamic'].shape).astype(bool)

        idx = 0
        if np.any(dynamic_objs):
            for semanticId in unique_semanticIds:
                # if we have processed all the dynamic objects
                if not np.any(dynamic_objs):
                    break

                inds = (instance_bboxes['semanticId'] == semanticId)
                final_inds = np.logical_and(dynamic_objs, inds)

                if np.any(final_inds):
                    label_name = id2label[semanticId].name
                    point_cloud_path = os.path.join(self.kitti360Path, 'data_3d_objects', f'{label_name}.npy')
                    if not os.path.exists(point_cloud_path):
                        dynamic_objs[final_inds] = False
                        continue
                    
                    #mesh = open3d.io.read_triangle_mesh(point_cloud_path)
                    #pcd = mesh.sample_points_uniformly(number_of_points=1000)

                    pcd_points = np.load(point_cloud_path).reshape((-1,3))

                    indices = self.cropPcd(pcd_points, num_points_per_obj)
                    pcd_points = pcd_points[indices,:]

                    indexes = np.where(final_inds)[0]
                    bboxes = instance_bboxes['bbox'][indexes,:]                     # nb_bboxes, 6
                    headings = instance_bboxes['heading'][indexes]                  # nb_bboxes
                    bboxes_instances_ids = instance_bboxes['instanceId'][indexes]   # nb_bboxes

                    #pcd_points = np.asarray(pcd.points)
                    nb_bboxes = bboxes.shape[0]
                    nb_points = pcd_points.shape[0]

                    color = id2label[semanticId].color
                    color = np.array(color).astype(np.float64)/255.0

                    new_points = np.tile(pcd_points, (nb_bboxes,1))                 # nb_bboxes * nb_points, 3
                    new_colors = np.tile(color, (new_points.shape[0],1))     # nb_bboxes * nb_points, 3
                    new_semanticIds = np.tile(semanticId, (new_points.shape[0]))    # nb_bboxes * nb_points,
                    new_instanceIds = np.tile(bboxes_instances_ids.reshape(nb_bboxes,1), (1,nb_points)).reshape((new_points.shape[0]))   # nb_bboxes * nb_points,

                    max_length = np.max(bboxes[:,-3:], axis=1)
                    scale = max_length / 2.
                    #scale = np.tile(scale, (1, nb_points)).reshape(new_points.shape[0]) # nb_bboxes * nb_points

                    new_points = new_points.reshape((nb_bboxes, nb_points, 3)) # nb_bboxes, N, 3
                    new_points = np.multiply(new_points, scale.reshape((nb_bboxes, 1, 1))) # nb_bboxes, nb_points, 3

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

                    if idx == 0:
                        points = np.copy(new_points)
                        colors = np.copy(new_colors)
                        semanticIds = np.copy(new_semanticIds)
                        instanceIds = np.copy(new_instanceIds)
                    else:
                        points = np.concatenate((points, new_points), axis=0)
                        colors = np.concatenate((colors, new_colors), axis=0)
                        semanticIds = np.concatenate((semanticIds, new_semanticIds), axis=0)
                        instanceIds = np.concatenate((instanceIds, new_instanceIds), axis=0)
                    
                    idx += 1

                    dynamic_objs[final_inds] = False
            
        return points, colors, semanticIds, instanceIds


    def cropPcd(self, points, nb_points):
        if points.shape[0] > nb_points:
            inds = np.random.choice(points.shape[0], nb_points, replace=False)            
        elif points.shape[0] > 0:
            inds = np.random.choice(points.shape[0], nb_points, replace=True)
        else:
            raise RuntimeError('[Kitti360BBoxesDataset] Empty point cloud')
        
        return inds


    def getFace(self, points, faces, face: str = 'front'):

        face = face.lower()
        if face == 'front':
            vertices_idx = self.front_vertices_idx
        elif face == 'back':
            vertices_idx = self.back_vertices_idx
        elif face == 'right':
            vertices_idx = self.right_vertices_idx
        elif face == 'left':
            vertices_idx = self.left_vertices_idx
        elif face == 'up':
            vertices_idx = self.up_vertices_idx
        elif face == 'down':
            vertices_idx = self.down_vertices_idx
        else:
            raise RuntimeError(f'[KITTI360Scene2Caption.getFace]: Unrecognised face {face}')

        face_vertices = points[vertices_idx]

        nb_faces, nb_vertices_per_face = faces.shape

        # Line
        if nb_vertices_per_face == 2:
            face_shape = 4
        # Triangle
        elif nb_vertices_per_face == 3:
            face_shape = 2
        else:
            raise RuntimeError('[getFace] Unrecognized face shape')
        
        new_faces = np.zeros((face_shape, nb_vertices_per_face))
        new_faces_idx = np.zeros(face_shape, dtype=np.int32)
        num_faces = 0
        for i in range(nb_faces):
            is_new_face = True
            for j in range(nb_vertices_per_face):
                if faces[i][j] not in vertices_idx:
                    is_new_face = False
                    break

            if is_new_face:
                new_faces[num_faces,:] = faces[i,:]
                new_faces_idx[num_faces] = i
                num_faces += 1

            if num_faces >= face_shape:
                break

        assert num_faces == face_shape, f'face doesn\'t have the required size {face_shape}'

        return face_vertices, new_faces, new_faces_idx


    def getColor(self, idx):
        if idx == 0:
            return np.array([0,0,0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3])*255.


    def assignColor(self, globalIds, gtType='semantic'):
        if not isinstance(globalIds, (np.ndarray, np.generic)):
            globalIds = np.array(globalIds)[None] # -- ME -- add a new axis
        color = np.zeros((globalIds.size, 3)) # -- ME -- (1, 4, 5) -> size = 4
        for uid in np.unique(globalIds):
            semanticId, instanceId = global2local(uid)
            if gtType == 'semantic':
                color[globalIds==uid] = id2label[semanticId].color
            elif instanceId > 0:
                color[globalIds==uid] = self.getColor(instanceId)
            else:
                color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
        color = color.astype(np.float64)/255.0
        return color


    def getLinesFromTriangles(self, triangles):
        triangles = triangles.astype(int)
        # count the number of occurrences of each vertice
        counter = {}
        for i in range(triangles.shape[0]):
            for j in range(triangles.shape[1]):
                if triangles[i,j] not in counter.keys():
                    counter[triangles[i,j]] = 0
                counter[triangles[i,j]] = counter[triangles[i,j]] + 1

        # convert dictionary to list of values sorted by key
        #counter_list = [counter[k] for k in sorted(counter)]

        #sort_idx = np.argsort(counter_list)
        lines = []
        for i in range(triangles.shape[0]):
            vertice_idx_min_occ = -1
            min_occ = triangles.shape[0] * 10
            for j in range(triangles.shape[1]):
                if counter[triangles[i,j]] < min_occ:
                    vertice_idx_min_occ = triangles[i,j]
                    min_occ = counter[triangles[i,j]]

            assert(vertice_idx_min_occ != -1)
                
            for j in range(triangles.shape[1]):
                if counter[triangles[i,j]] > min_occ:
                    lines.append([vertice_idx_min_occ, triangles[i,j]])

        lines = np.asarray(lines).reshape(-1, 2)
        lines = np.unique(lines, axis=0)
        return lines
    

def getValidLabels(labels: List = None, categories: List = None):
    # get the categories labels
    if (categories is not None) and isinstance(categories, List):
        categories = list(set([label.category for label in LABELS if label.category in categories]))
    else:
        categories = []

    # get the labels and remove categories already used
    valid_labels = []
    if (labels is not None) and isinstance(labels, List):
        for label in LABELS:
            if label.name in labels:
                valid_labels.append(label.name)

                if len(categories) > 0 and (label.category in categories):
                    categories.remove(label.category)

    # complete the selected labels with the labels of the remaining categories
    if len(categories) > 0:
        for category in categories:
            for label in category2labels[category]:
                valid_labels.append(label.name)

    return valid_labels


def process_one_sequence(config: Dict):
    seq2cap = KITTI360Maps(config)
    nb_scene_id = seq2cap.run()

    return nb_scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sequences', type=str, default='0', help='The sequence to process')   
    parser.add_argument('-c', '--categories', type=str, default='vehicle,human', help='The categories you wan to caption')
    parser.add_argument('-l', '--labels', type=str, default='', help='The labels you want to caption')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('--num_points', type=int, default=100000, help='Number of points to save')
    parser.add_argument('--showStatic', type=lambda x: (str(x).lower() == 'true'), default=True,
                                help='Whether you want to consider only static objects')
    parser.add_argument('--withGround', type=lambda x: (str(x).lower() == 'true'), default=False, 
                                help='Whether to process \'train\' or \'train_full\' xml files')

    args = parser.parse_args()

    sequences_str = args.sequences
    sequences = [int(s) for s in sequences_str.split(',')]

    categories_str = args.categories
    categories = categories_str.split(',')

    labels_str = args.labels
    labels = labels_str.split(',')

    filtering_labels = getValidLabels(labels=labels, categories=categories)

    config = dict()
    config['labels'] = filtering_labels
    config['showStatic'] = args.showStatic
    config['withGround'] = args.withGround
    config['output_dir'] = args.output_dir
    config['num_points'] = args.num_points

    avg_seq_elapsed_time = 0.
    for i, seq in enumerate(sequences):
        config['sequence'] = seq
        start = time.time()

        process_one_sequence(config)

        if i < (len(sequences) - 1):
            end = time.time()
            elapsed_time = end -start
            avg_seq_elapsed_time = (avg_seq_elapsed_time * i + elapsed_time) / (i+1)
            print('Approximate remaining time is:', avg_seq_elapsed_time*(len(sequences)-(i+1)), '\n')

if __name__ == '__main__':
    main()
