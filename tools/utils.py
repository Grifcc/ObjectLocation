from typing import Union
import trimesh
import mesh_raycast
import numpy as np
from enum import Enum


class PointType(Enum):
    ValidPoint = 0  # 有效点
    NonePoint = 1  # 无交点
    Obscured = 2  # 遮挡
    OOF = 3   # 视场外 out of FOV
    OOC = 4  # 图像外 out of camera
    Other = -1  # 其他


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

    def _get_cxcywh(self, point: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0], point[1]-self.height/2, self.width, self.height]
        else:
            return PointType.OOC

    def _get_x1y1x2y2(self, point: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0]-self.width/2, point[1]-self.height, point[0]+self.width/2, point[1]]
        else:
            return PointType.OOC

    def _get_xywh(self, point: Union[tuple, list]):
        if self.valid_point(point):
            return [point[0]-self.width/2, point[1]-self.height, point[0], point[1]]
        else:
            return PointType.OOC

    def valid_point(self, point: Union[tuple, list]):
        if point[0] < 0 or point[1] < 0 or point[0] >= self.width or point[1] >= self.height:
            return False
        else:
            return True


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


class SimulationCamera:
    def __init__(self, poses, camera_K, distortion_coeffs, mesh_path, img_shape, threshold=1.5, bbox_type='cxcywh'):
        self.K = camera_K
        self.distort = distortion_coeffs
        self.camera_K, self.camera_K_inv = set_K(camera_K)
        self.distortion_coeffs = set_distortion_coeffs(distortion_coeffs)
        self.mesh = self.read_mesh(mesh_path)
        self.poses = self.create_pose(poses)
        self.img = ImageView(img_shape, bbox_type)

        self.threshold = threshold  # 判断是否被遮挡的阈值

    def create_pose(self, poses):
        rotation_matrix_i, translation_vector_i = set_camera_pose(poses)
        camera_rotation_inv_i = np.linalg.inv(rotation_matrix_i)
        self.rotation_matrix.append(rotation_matrix_i)
        self.translation_vector.append(translation_vector_i)
        self.camera_rotation_inv.append(camera_rotation_inv_i)

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

    # gt_gis_point[x,y] to  bbox   空间三维点到像平面的二维点
    def get_bbox_result(self, gt_point) -> list:

        # 根据 xy找z
        ray_origins = np.array([gt_point[0], gt_point[1], 200]).reshape(3, 1)
        ray_directions = [0, 0, -1.]
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            return PointType.NonePoint, []  # 地图空洞，无交点

        result_point = min(result, key=lambda x: x['distance'])[
            'point']  # 地图有交点 TODO: 需不需要判断cos值，防止与三角面的法向量相反，交在背面

        p_c1 = self.camera_rotation_inv@(result_point-self.translation_vector)
        p_cd = p_c1/p_c1[2]
        pixel = self.camera_K@p_cd

        # 正向求解
        pixel = [pixel[0][0], pixel[1][0]]
        bbox = self.img.get_bbox(pixel)
        if bbox == PointType.OOC:
            return PointType.OOC, ()  # 图像外

        p_c1 = undistort_pixel_coords(
            pixel, self.camera_K_inv, self.distortion_coeffs)
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(self.rotation_matrix, pc_d)
        ray_origins = self.translation_vector
        ray_directions = pw_d.flatten()
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            return PointType.NonePoint, ()  # 地图空洞，无交点

        pred_point = min(result, key=lambda x: x['distance'])[
            'point']  # 地图有交点 TODO: 需不需要判断cos值，防止与三角面的法向量相反，交在背面

        if pred_point[0] - gt_point[0] > self.threshold or pred_point[1] - gt_point[1] > self.threshold:  # 遮挡
            return PointType.Obscured, ()
        else:
            return PointType.ValidPoint, (bbox, result_point, pred_point)
