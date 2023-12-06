import json
import os
import trimesh
import mesh_raycast
import numpy as np
import time
import math

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
    def __init__(self, poses, camera_K, distortion_coeffs, mesh_path):
        self.K = camera_K
        self.distort = distortion_coeffs
        self.camera_K, self.camera_K_inv = set_K(camera_K)
        self.distortion_coeffs = set_distortion_coeffs(distortion_coeffs)
        self.mesh = self.read_mesh(mesh_path)
        self.rotation_matrix = []
        self.camera_rotation_inv = []
        self.translation_vector = []
        self.now_xy = None
        self.now_timestamp  = None
        self.class_id  = None
        self.track_id = [None, None, None]
        self.obj_size = None
        self.real_point = None # list:[x,y,z]
        self.poses = self.create_pose(poses)

    def create_pose(self, poses):
        for i in range(len(poses)):
            rotation_matrix_i, translation_vector_i = set_camera_pose(poses[i])
            camera_rotation_inv_i = np.linalg.inv(rotation_matrix_i)
            self.rotation_matrix.append(rotation_matrix_i)
            self.translation_vector.append(translation_vector_i)
            self.camera_rotation_inv.append(camera_rotation_inv_i)
        return poses

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
    
    def start_set(self, now_timestamp, now_xy, class_id, obj_size):
        self.real_point = None
        self.now_timestamp = now_timestamp
        self.now_xy = now_xy
        self.class_id = class_id
        self.obj_size = obj_size

    def track_reset(self):
        self.track_id = [None, None, None]


    def get_real_z(self):
        ray_origins = np.array([self.now_xy[0], self.now_xy[1], 200]).reshape(3, 1)
        ray_directions = [0, 0, -1.]
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            self.real_point = [0.0, 0.0, 0.0]
            return False
        else:
            first_result = min(result, key=lambda x: x['distance'])
            self.real_point = list(first_result['point'])
            return True

    def distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

    def InCamera(self, piexl, pred_point):
        if piexl[0] <0 or piexl[1]<0 or piexl[0]>=640 or piexl[1]>=480:
            return False
        elif self.distance(pred_point, self.real_point)>1:
            return False
        else:
            return True

    def get_pixel_result(self, track_id):
        real_point = np.array(self.real_point).reshape(3, 1)
        for i in range(len(self.rotation_matrix)): # 第i个相机

            p_c1 = self.camera_rotation_inv[i]@(real_point-self.translation_vector[i])
            p_cd = p_c1/p_c1[2]
            pixel = self.camera_K@p_cd
            # [cx,cy,w,h]
            img_pixel = [pixel[0][0]-self.obj_size[0]/2., pixel[1][0]-self.obj_size[1], self.obj_size[0], self.obj_size[1]]
            # 找到预测点
            pixel = [pixel[0][0], pixel[1][0]]
            p_c1 = undistort_pixel_coords(pixel, self.camera_K_inv, self.distortion_coeffs)
            pc_d = p_c1 / np.linalg.norm(p_c1)
            pw_d = np.dot(self.rotation_matrix[i], pc_d)
            ray_origins = self.translation_vector[i]
            ray_directions = pw_d.flatten()
            result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
            # z是否有交点，否则退出
            if len(result) == 0:  
                result_point = np.array([0., 0, 0]).reshape(3, 1)
                continue
            else:
                first_result = min(result, key=lambda x: x['distance'])
                result_point = first_result['point']
                pred_point = list(result_point)
            
            # pixel是否超出分辨率范围或者遮挡导致pred和real不同
            if not self.InCamera(img_pixel, pred_point):
                continue
            else:
                # 如果该目标被该相机追踪过，那track_id不变
                if self.track_id[i] is not None:
                    pass
                else: #没被追踪过给id，加一为了后续目标 
                    
                    self.track_id[i] = track_id[i]
                    track_id[i] = track_id[i] + 1 # （因为其他目标被追踪，id+1）TODO这个需要保持一致，或者return

                print((f"time: {self.now_timestamp}, "
                    f"uav_id: {i}, "
                    f"camera_pose: {self.poses[i]}, "
                    # f"camera_K: {self.K}, "
                    # f"camera_distortion: {self.distort}, "
                    f"Bbox: {img_pixel}, "
                    f"class_id: {self.class_id}, "
                    f"tracker_id: {self.track_id[i]}, "
                    ))


                # # TODO 这里写入json，你可以复用之前写的代码,这里是伪代码
                # self.time = self.now_timestamp   
                # self.uav_id: int =  i
                # self.camera_pose: list[float] = self.poses[i]  # [yaw,pitch,roll,x,y,z]
                # self.camera_K: list[float] = self.K  # [fx,fy,cx,cy]
                # self.camera_distortion: list[float] = self.distort # [k1,k2,p1,p2]
                # self.Bbox: list[int] = img_pixel  # [x,y,w,h]
                # self.class_id: int = self.class_id # 0人1车
                # self.class_name: str = None
                # self.tracker_id: int = self.track_id[i]
                # self.uav_pos: list[float] = []
                # self.obj_img: str = None
                # # read & write
                # self.global_id: int = None
                # self.local_id: int = None
                # self.location: list[float] = []  # (WGS84）
