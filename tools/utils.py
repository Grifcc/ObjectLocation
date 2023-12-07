import json
import os
import trimesh
import mesh_raycast
import numpy as np
import time


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


def get_rays_corners(H, W, K, R, t):
    lt = np.array([[0., 0, 1]], dtype=np.float32).reshape(3,1)
    rt = np.array([[W-1, 0, 1.]], dtype=np.float32).reshape(3,1)
    rd = np.array([[W-1, H-1, 1.]], dtype=np.float32).reshape(3,1)
    ld = np.array([[0., H-1, 1.]], dtype=np.float32).reshape(3,1)
 
    # 定义图像的四个角点坐标（左上、右上、右下、左下）
    uvs = [lt, rt, rd, ld]
    rays_o = [] #射线起点，其实都一样 TODO
    rays_d = [] # 射线方向
    for uv in uvs:
        p_cam = np.linalg.inv(K) @ uv
        p_world = R @ p_cam
        ray_o = t
        ray_d = p_world / np.linalg.norm(p_world)
        rays_o.append(ray_o)
        rays_d.append(ray_d)
    return rays_o, rays_d


def compute_xy_coordinate(rays_o, rays_d):
    inter_points = []
    for i in range(4):
        # 计算射线与XY平面的交点的t值 o+td = 0
        t = -rays_o[i][2] / rays_d[i][2]

        # 计算交点坐标
        inter_point = rays_o[i] + t * rays_d[i]
        inter_points.append(inter_point.flatten())

    return inter_points


class SimulationCamera:
    def __init__(self, camera_pose, camera_K, distortion_coeffs, mesh_path):
        self.camera_pose = camera_pose
        self.camera_K, self.camera_K_inv = set_K(camera_K)

        self.rotation_matrix, self.translation_vector = set_camera_pose(
            self.camera_pose)
        self.camera_rotation_inv = np.linalg.inv(self.rotation_matrix)

        self.distortion_coeffs = set_distortion_coeffs(distortion_coeffs)

        self.mesh = self.read_mesh(mesh_path)

    def read_mesh(self, mesh_path):
        mesh = trimesh.load_mesh(mesh_path)
        all_meshes = [geom for geom in mesh.geometry.values()]
        # 使用 concatenate 函数将多个 mesh 合并为一个
        combined_mesh = trimesh.util.concatenate(all_meshes)
        vertices = combined_mesh.vertices
        faces = combined_mesh.faces
        triangles = vertices[faces]
        triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
        mesh = triangles
        return triangles

    def generate_simulation(self, point, obj_size: list):
        ray_origins = np.array([point[0], point[1], 200]).reshape(3, 1)
        ray_directions = [0, 0, -1.]
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            result_point = [0.0, 0.0, 0.0]
        else:
            first_result = min(result, key=lambda x: x['distance'])
            result_point = first_result['point']

        result_point = list(result_point)
        real_point = result_point[:]

        result_point = np.array(result_point).reshape(3, 1)
        p_c1 = self.camera_rotation_inv@(result_point-self.translation_vector)
        p_cd = p_c1/p_c1[2]
        pixel = self.camera_K@p_cd

        # [cx,cy,w,h]
        img_pixel = [pixel[0][0]-obj_size[0]/2., pixel[1][0]-obj_size[1], obj_size[0], obj_size[1]]
        # 找到预测点
        pixel = [pixel[0][0], pixel[1][0]]
        p_c1 = undistort_pixel_coords(
            pixel, self.camera_K_inv, self.distortion_coeffs)
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(self.rotation_matrix, pc_d)
        ray_origins = self.translation_vector
        ray_directions = pw_d.flatten()
        result = mesh_raycast.raycast(ray_origins, ray_directions, self.mesh)
        if len(result) == 0:  # TODO 可优化
            result_point = np.array([0., 0, 0]).reshape(3, 1)
        else:
            first_result = min(result, key=lambda x: x['distance'])
            result_point = first_result['point']

        pred_point = list(result_point)

        return real_point, img_pixel, pred_point
