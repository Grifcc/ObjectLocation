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


def set_camera_pose(camera_pose):  # (yaw,pitch,row,x,y,z)

    roll = camera_pose[2]   # roll
    pitch = camera_pose[1]   # pitch
    yaw = camera_pose[0]   # yaw

    # 角度转弧度
    roll = np.radians(roll)  # 绕X轴旋转
    pitch = np.radians(pitch)  # 绕Y轴旋转
    yaw = np.radians(yaw)  # 绕Z轴旋转

    t = np.array(camera_pose[3:], dtype=float)
    # 构建绕X、Y、Z轴旋转的矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]])

    # 得到总的旋转矩阵
    R = R_z @ R_y @ R_x
    return R, t.reshape(3, 1), np.linalg.inv(R)


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


def compute_xy_coordinate(rays_o: list, rays_d: list, height=0):
    """
    计算光线与XY平面的交点坐标。

    Args:
        rays_o : 光线起点列表。
        rays_d (list): 光线方向列表。
        height (float): 平面的高度,默认为0。

    Returns:
        list: 包含光线与XY平面的交点坐标的列表。
    """
    inter_points = []
    if not isinstance(rays_o, np.ndarray):
        rays_d = np.array(rays_d, dtype=np.float32)
    for i in range(4):
        # 计算射线与XY平面的交点的t值 o+td = 0
        t = (height - rays_o[2]) / rays_d[i][2]

        # 计算交点坐标
        inter_point = rays_o + t * rays_d[i]
        inter_points.append(inter_point.flatten().tolist())

    return inter_points


def get_real_point(point, mesh):
    # 根据 xy找z
    ray_origins = np.array([point[0], point[1], 200]).reshape(3, 1)
    ray_directions = [0, 0, -1.]
    result = mesh_raycast.raycast(ray_origins, ray_directions, mesh)
    if len(result) == 0:  # TODO 可优化
        return PointType.NonePoint, ()  # 地图空洞,无交点
    result_point = min(result, key=lambda x: x['distance'])[
        'point']  # 地图有交点 TODO: 需不需要判断cos值,防止与三角面的法向量相反,交在背面
    return PointType.ValidPoint, result_point


def read_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    all_meshes = [geom for geom in mesh.geometry.values()]
    # 使用 concatenate 函数将多个 mesh 合并为一个
    combined_mesh = trimesh.util.concatenate(all_meshes)
    vertices = combined_mesh.vertices
    faces = combined_mesh.faces
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')  # 一定要有这一行,不然会有错。
    return triangles


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
    def __init__(self, obj_attr: dict, num_view: int = 1, max_age: int = 10, uid=None):
        self.start_point = obj_attr["start_point"]
        self.speed = obj_attr["speed"]
        self.class_id = obj_attr["cls_id"]
        self.move_radians = obj_attr["angle"] * math.pi / 180
        self.now_point = self.start_point[:]
        self.bbox_size = obj_attr["bbox"]
        self.uid = uid if uid != None else id(self)
        self.tracker_id = [-1] * num_view
        self.max_age = max_age
        self.age = [0] * num_view

    def next_point(self):
        self.now_point[0] += self.speed*np.cos(self.move_radians)
        self.now_point[1] += self.speed*np.sin(self.move_radians)
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
    def __init__(self, poses: list[float], camera_K: list[float], distortion_coeffs: list[float], img_shape: Union[tuple, list], mesh_path: str = None, threshold: float = 1.5, bbox_type='cxcywh'):

        self.pose = poses
        self.camera_K, self.camera_K_inv = set_K(camera_K)
        self.distortion_coeffs = set_distortion_coeffs(distortion_coeffs)
        self.rotation_matrix, self.translation_vector, self.camera_rotation_inv = set_camera_pose(
            self.pose)
        self.mesh = read_mesh(mesh_path) if mesh_path != None else None
        self.img_shape = img_shape
        self.img = ImageView(self.img_shape, bbox_type)

        self.threshold = threshold  # 判断是否被遮挡的阈值

        self.max_id = 0

    def get_params(self):
        """
        获取相机参数。

        Returns:
            tuple: 包含相机内参、旋转矩阵、畸变系数和平移向量的元组。
        """
        return self.camera_K, self.rotation_matrix, self.distortion_coeffs, self.translation_vector.flatten()

    def get_max_id(self):
        return self.max_id

    def get_rays_corners(self):
        """
        计算光线的角点坐标。
        根据:
            H (int): 图像的高度。
            W (int): 图像的宽度。
            K (np.ndarray): 内参矩阵。
            R (np.ndarray): 旋转矩阵。
            t (np.ndarray): 平移向量。

        Returns:
            rays_d(list): 相机坐标系下的射线方向。 
        """
        lt = np.array([[0., 0, 1]], dtype=np.float32).reshape(3, 1)
        rt = np.array([[self.img_shape[0]-1, 0, 1.]],
                      dtype=np.float32).reshape(3, 1)
        rd = np.array([[self.img_shape[0]-1, self.img_shape[1]-1, 1.]],
                      dtype=np.float32).reshape(3, 1)
        ld = np.array([[0., self.img_shape[1]-1, 1.]],
                      dtype=np.float32).reshape(3, 1)

        # 定义图像的四个角点坐标（左上、右上、右下、左下）
        uvs = [lt, rt, rd, ld]
        rays_d = []  # 射线方向
        for uv in uvs:
            p_cam = self.camera_K_inv @ uv
            p_world = self.rotation_matrix @ p_cam
            ray_d = p_world / np.linalg.norm(p_world)
            rays_d.append(ray_d.tolist())
        return rays_d
    
    def get_fov_angle(self):
        """
        计算相机的视场角。

        Returns:
            list[float]: 相机的视场角[angle_h,angle_v]。
        """

        rays_d = self.get_rays_corners()
        if not isinstance(rays_d, np.ndarray):
            rays_d = np.array(rays_d, dtype=np.float32).reshape(4, 3)
        
        rad_h = math.acos(np.dot(rays_d[0], rays_d[1]))
        rad_v = math.acos(np.dot(rays_d[1], rays_d[3]))

        return math.degrees(rad_h),math.degrees(rad_v)
    

    def get_fov_scope(self, height=0):
        return compute_xy_coordinate(self.translation_vector.flatten().tolist(), self.get_rays_corners(), height)

    def pixel2point(self, pixel):
        """
        将像素坐标转换为三维点坐标。

        参数:
        pixel:像素坐标,形如 [x_pixel, y_pixel]

        返回值:
        如果地图为空洞,返回 PointType.NonePoint 和空元组 ()
        如果地图有交点,返回 PointType.ValidPoint 和交点坐标 result_point

        """
        assert not self.mesh is None, "mesh is None"
        p_c1 = undistort_pixel_coords(
            pixel, self.camera_K_inv, self.distortion_coeffs)
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(self.rotation_matrix, pc_d)
        ray_origins = self.translation_vector
        ray_directions = pw_d.flatten()
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            return PointType.NonePoint, ()  # 地图空洞,无交点
        result_point = min(result, key=lambda x: x['distance'])[
            'point']  # 地图有交点 TODO: 需不需要判断cos值,防止与三角面的法向量相反,交在背面
        return PointType.ValidPoint, result_point

    def point2pixel(self, point):
        """
        将三维点坐标转换为像素坐标。

        参数:
        point:三维点的坐标,形如 [x, y, z]

        返回值:
        像素坐标,形如 [x_pixel, y_pixel]

        """
        point = np.array(point).reshape(3, 1)
        p_c1 = self.camera_rotation_inv@(point-self.translation_vector)
        p_cd = p_c1/p_c1[2]
        pixel = self.camera_K@p_cd
        return [pixel[0][0], pixel[1][0]]

    def get_bbox_result(self, gt_point, bbox_size: Union[tuple, list]):
        """
        将空间三维点转换为像平面的二维点,并返回对应的边界框结果。

        参数:
        gt_point:空间三维点的坐标,形如 [x, y, z]
        bbox_size:边界框的尺寸,形如 (width, height)

        返回值:
        如果地图为空洞,返回 PointType.NonePoint 和空元组 ()
        如果图像外,返回 PointType.OOC 和空元组 ()
        如果被遮挡,返回 PointType.Obscured 和空元组 ()
        如果有效,返回 PointType.ValidPoint 和包含边界框、真实点坐标和预测点坐标的元组 (bbox, result_point, pred_point)

        """
        assert not self.mesh is None, "mesh is None"
        # 根据 xy 找到对应的真实点坐标
        status, result_point = get_real_point(gt_point, self.mesh)
        if status != PointType.ValidPoint:
            return status, (None, None, None)

        # 逆向求解,将真实点坐标转换为像平面的二维点
        pixel = self.point2pixel(result_point)

        # 判断像平面的二维点是否在图像内
        bbox = self.img.get_bbox(pixel, bbox_size)
        if bbox == PointType.OOC:
            return PointType.OOC, (None, result_point, None)  # 图像外

        # 正向求解,将像平面的二维点转换为空间三维点
        status, pred_point = self.pixel2point(pixel)
        if status != PointType.ValidPoint:
            return status, (None, None, None)

        # 判断是否被遮挡
        if np.linalg.norm(np.array(pred_point)-np.array(result_point)) > self.threshold:
            return PointType.Obscured, (None, result_point, None)
        else:
            return PointType.ValidPoint, (bbox, result_point, pred_point)
