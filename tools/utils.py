from typing import Union
import trimesh
import mesh_raycast
import numpy as np
from enum import Enum
import math


class PointType(Enum):
    ValidPoint = 0  # 有效点
    NonePoint = 1  # 无交点
    Obscured = 2  # 遮挡
    OOF = 3   # 视场外 out of FOV
    OOC = 4  # 图像外 out of camera
    Other = -1  # 其他


def set_K(cam_K):
    fx, fy, cx, cy = cam_K
    # 构建内参矩阵
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    K_inv = np.linalg.inv(K)
    return K, K_inv


def set_distortion_coeffs(distortion_param):
    k1, k2, k3, p1, p2 = distortion_param
    return np.array([k1, k2, k3, p1, p2])


def set_rotation_matrix(t1, t2, t3):  # (roll, pitch, yaw)
    # 角度转弧度
    theta1 = np.radians(t1)  # 绕X轴旋转
    theta2 = np.radians(t2)  # 绕Y轴旋转
    theta3 = np.radians(t3)  # 绕Z轴旋转

    # 构建绕X、Y、Z轴旋转的矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta1), -np.sin(theta1)],
        [0, np.sin(theta1), np.cos(theta1)]
    ])

    R_y = np.array([
        [np.cos(theta2), 0, np.sin(theta2)],
        [0, 1, 0],
        [-np.sin(theta2), 0, np.cos(theta2)]
    ])

    R_z = np.array([
        [np.cos(theta3), -np.sin(theta3), 0],
        [np.sin(theta3), np.cos(theta3), 0],
        [0, 0, 1]
    ])

    # 得到总的旋转矩阵
    rotation_matrix = R_z @ R_y @ R_x
    # rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    return rotation_matrix


def set_translation_vector(t1, t2, t3):
    return np.array([[t1], [t2], [t3]]).reshape(3, 1)


def set_camera_pose(camera_pose):
    rotation_matrix = set_rotation_matrix(
        t1=camera_pose[2], t2=camera_pose[1], t3=camera_pose[0])
    translation_vector = np.array(
        [[camera_pose[3]], [camera_pose[4]], [camera_pose[5]]]).reshape(3, 1)
    return rotation_matrix, translation_vector


def undistort_pixel_coords(pixel_coords, camera_K_inv, distortion_coeffs):
    # 像素坐标转为齐次坐标
    pixel = np.array([pixel_coords[0], pixel_coords[1], 1.]).reshape(3, 1)
    # 像素坐标转换到相机坐标系下
    p_cam = np.dot(camera_K_inv, pixel)
    # 畸变校正的计算过程
    x = p_cam[0][0]
    y = p_cam[1][0]
    r_sq = np.sqrt(x**2 + y**2)
    x_correction = x * (1 + distortion_coeffs[0] * r_sq + distortion_coeffs[1] * r_sq**2 + distortion_coeffs[2] * r_sq**3) + (
        2 * distortion_coeffs[3] * x * y + distortion_coeffs[4] * (r_sq**2 + 2 * x**2))
    y_correction = y * (1 + distortion_coeffs[0] * r_sq + distortion_coeffs[1] * r_sq**2 + distortion_coeffs[2] * r_sq**3) + (
        distortion_coeffs[3] * (r_sq**2 + 2 * y**2) + 2 * distortion_coeffs[4] * x * y)
    # 校正后的相机坐标
    p_cam_distorted = np.array([x_correction, y_correction, 1.]).reshape(3, 1)
    return p_cam_distorted


class ImageView:
    def __init__(self, shape: Union[tuple, list], bbox_type='cxcywh'):
        self.shape = shape
        self.width = shape[0]
        self.height = shape[1]
        self.bbox_type = bbox_type
        self.get_bbox = None
        if bbox_type == 'cxcywh':
            self.get_bbox = self._get_cxcywh
        elif bbox_type == 'x1y1x2y2':
            self.get_bbox = self._get_x1y1x2y2
        elif bbox_type == 'xywh':
            self.get_bbox = self._get_xywh
        else:
            raise ValueError("bbox_type must be 'cxcywh' or 'x1y1x2y2'")

    def _get_cxcywh(self, point: Union[tuple, list], bbox_size: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0], point[1] - bbox_size[1]/2, bbox_size[0], bbox_size[1]]
        else:
            return PointType.OOC

    def _get_x1y1x2y2(self, point: Union[tuple, list], bbox_size: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0]-bbox_size[0], point[1]-bbox_size[1], point[0]+bbox_size[0]/2, bbox_size[1]]
        else:
            return PointType.OOC

    def _get_xywh(self, point: Union[tuple, list], bbox_size: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0] - bbox_size[0]/2, point[1]-bbox_size[1], bbox_size[0], bbox_size[1]]
        else:
            return PointType.OOC

    def valid_point(self, point: Union[tuple, list]):
        if point[0] < 0 or point[1] < 0 or point[0] >= self.width or point[1] >= self.height:
            return False
        else:
            return True


class SimulationObject:
    def __init__(self, start_point, speed, bbox_size, class_id, move_angle, num_view, max_age=10, uid=None):
        self.start_point = start_point[:]
        self.speed = speed
        self.class_id = class_id
        self.move_angle = move_angle * math.pi / 180
        self.now_point = start_point[:]
        self.bbox_size = bbox_size[:]
        self.uid = uid if uid else id(self)
        self.tracker_id = [-1] * num_view
        self.max_age = max_age
        self.age = [0] * num_view

    def next_point(self):
        self.now_point[0] += self.speed*np.cos(self.move_angle)
        self.now_point[1] += self.speed*np.sin(self.move_angle)
        return self.now_point

    def get_clsid(self):
        return self.class_id

    def get_bbox_size(self):
        return self.bbox_size

    def set_speed(self, speed):
        self.speed = speed

    def reset_age(self, view_id):
        self.age[view_id] = 0

    def __str__(self) -> str:
        return f"The Object:\n\tuid:{self.uid}\n\tclass_id:{self.class_id}\n\tspeed:{self.speed}\n\ttracker_id:{self.tracker_id}\n\tage:{self.age}\n\tnow_point:{self.now_point}\n\tbbox_size:{self.bbox_size}\n\tmove_angle:{self.move_angle}\n"


class SimulationCamera:
    def __init__(self, poses, camera_K, distortion_coeffs, mesh_path, img_shape, threshold=1.5, bbox_type='cxcywh'):

        self.pose = poses
        self.K = camera_K
        self.distort = distortion_coeffs
        self.camera_K, self.camera_K_inv = set_K(camera_K)
        self.distortion_coeffs = set_distortion_coeffs(distortion_coeffs)
        self.mesh = self.read_mesh(mesh_path)
        self.rotation_matrix, self.translation_vector, self.camera_rotation_inv = self.create_pose(
            self.pose)

        self.img_shape = img_shape
        self.img = ImageView(self.img_shape, bbox_type)

        self.threshold = threshold  # 判断是否被遮挡的阈值

        self.max_id = 0

    def get_params(self):
        return self.pose, self.K, self.distort

    def get_max_id(self):
        return self.max_id

    def create_pose(self, poses):
        rotation_matrix_i, translation_vector_i = set_camera_pose(poses)
        camera_rotation_inv_i = np.linalg.inv(rotation_matrix_i)
        return rotation_matrix_i, translation_vector_i, camera_rotation_inv_i

    def read_mesh(self, mesh_path):
        mesh = trimesh.load_mesh(mesh_path)
        all_meshes = [geom for geom in mesh.geometry.values()]
        # 使用 concatenate 函数将多个 mesh 合并为一个
        combined_mesh = trimesh.util.concatenate(all_meshes)
        vertices = combined_mesh.vertices
        faces = combined_mesh.faces
        triangles = vertices[faces]
        triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
        return triangles

    def get_scope(self, height=0):
        W, H = self.img_shape
        lt = np.array([[0., 0, 1]], dtype=np.float32).reshape(3, 1)
        rt = np.array([[W-1, 0, 1.]], dtype=np.float32).reshape(3, 1)
        rd = np.array([[W-1, H-1, 1.]], dtype=np.float32).reshape(3, 1)
        ld = np.array([[0., H-1, 1.]], dtype=np.float32).reshape(3, 1)
        uvs = [lt, rt, rd, ld]  # 左上、右上、右下、左下
        inter_points = []
        for uv in uvs:
            p_cam = np.linalg.inv(self.camera_K) @ uv
            p_world = self.rotation_matrix @ p_cam
            ray_o = self.translation_vector
            ray_d = p_world / np.linalg.norm(p_world)
            ray_d = ray_d.flatten()
            t = (height-ray_o[2])/ray_d[2]
            inter_point = ray_o + t * ray_d
            inter_points.append(inter_point.flatten())
        return inter_points

    def get_z(self, point):
        # 根据 xy找z
        ray_origins = np.array([point[0], point[1], 200]).reshape(3, 1)
        ray_directions = [0, 0, -1.]
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            return PointType.NonePoint, ()  # 地图空洞，无交点
        result_point = min(result, key=lambda x: x['distance'])[
            'point']  # 地图有交点 TODO: 需不需要判断cos值，防止与三角面的法向量相反，交在背面
        return PointType.ValidPoint, result_point

    def pixel2point(self, pixel):
        p_c1 = undistort_pixel_coords(
            pixel, self.camera_K_inv, self.distortion_coeffs)
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(self.rotation_matrix, pc_d)
        ray_origins = self.translation_vector
        ray_directions = pw_d.flatten()
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            return PointType.NonePoint, ()  # 地图空洞，无交点
        result_point = min(result, key=lambda x: x['distance'])[
            'point']  # 地图有交点 TODO: 需不需要判断cos值，防止与三角面的法向量相反，交在背面
        return PointType.ValidPoint, result_point

    def point2pixel(self, point):
        p_c1 = self.camera_rotation_inv@(point-self.translation_vector)
        p_cd = p_c1/p_c1[2]
        pixel = self.camera_K@p_cd
        return [pixel[0][0], pixel[1][0]]

    # gt_gis_point[x,y] to  bbox   空间三维点到像平面的二维点
    def get_bbox_result(self, gt_point, bbox_size: Union[tuple, list]):

        # 根据 xy找z
        status, result_point = self.get_z(gt_point)
        if status != PointType.ValidPoint:
            return status, ()

        # 逆向求解
        pixel = self.point2pixel(result_point)

        # 判断是否在图像内
        bbox = self.img.get_bbox(pixel, bbox_size)
        if bbox == PointType.OOC:
            return PointType.OOC, ()  # 图像外

        # 正向求解
        status, pred_point = self.pixel2point(pixel)
        if status != PointType.ValidPoint:
            return status, ()

        # 判断是否被遮挡
        if np.linalg.norm(pred_point-result_point) > self.threshold:
            return PointType.Obscured, ()
        else:
            return PointType.ValidPoint, (bbox, result_point, pred_point)
