from framework import Package
from framework import Location
import trimesh
import numpy as np
import mesh_raycast
import trimesh 
import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation

class EstiPosition(Location):
    def __init__(self, mesh_path=None):
        super().__init__("EstiPosition",)
        self.mesh = self.set_Mesh(mesh_path)  # mesh地图

    # TODO 这个对于不同的mesh地图有error
    def set_Mesh(self, path):
        mesh = trimesh.load_mesh(path) #load or load_mesh
        all_meshes = [geom for geom in mesh.geometry.values()]
        # 使用 concatenate 函数将多个 mesh 合并为一个
        combined_mesh = trimesh.util.concatenate(all_meshes)
        vertices = combined_mesh.vertices
        faces = combined_mesh.faces
        triangles = vertices[faces]
        triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
        return triangles

    def set_K(self, cam_K):
        fx,fy,cx,cy = cam_K
        # 构建内参矩阵
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        K_inv = np.linalg.inv(K)
        return K, K_inv
    
    def set_distortion_coeffs(self,  distortion_param):
        k1, k2, k3, p1, p2 = distortion_param
        return np.array([k1, k2,k3, p1, p2])

    def set_rotation_matrix(self, t1, t2, t3): # (roll, pitch, yaw)
        # 角度转弧度
        theta1 = np.radians(t1) # 绕X轴旋转 
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
    
    def set_translation_vector(self, t1, t2, t3):
        return np.array([[t1], [t2], [t3]]).reshape(3,1)

    def set_camera_pose(self, camera_pose):
        rotation_matrix = self.set_rotation_matrix(t1=camera_pose[2],t2=camera_pose[1],t3=camera_pose[0])
        translation_vector = np.array([[camera_pose[3]], [camera_pose[4]], [camera_pose[5]]]).reshape(3,1)
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

        x_correction = x * (1 + distortion_coeffs[0] * r_sq + distortion_coeffs[1] * r_sq**2 + distortion_coeffs[2] * r_sq**3) + (2 * distortion_coeffs[3] * x * y + distortion_coeffs[4] * (r_sq**2 + 2 * x**2))
        y_correction = y * (1 + distortion_coeffs[0] * r_sq + distortion_coeffs[1] * r_sq**2 + distortion_coeffs[2] * r_sq**3) + (distortion_coeffs[3] * (r_sq**2 + 2 * y**2) + 2 * distortion_coeffs[4] * x * y)
        # 校正后的相机坐标
        p_cam_distorted = np.array([x_correction, y_correction, 1.]).reshape(3,1)
        return p_cam_distorted


    def get_point(self, data: Package):
        camera_K, camera_K_inv = self.set_K(cam_K=data.camera_K)
        distortion_coeffs = self.set_distortion_coeffs(data.camera_distortion)
        rotation_matrix, translation_vector = self.set_camera_pose(camera_pose=data.camera_pose)

        # 获取像素并去畸变
        pixel = data.get_center_point() # [float, float]
        p_c1 = self.undistort_pixel_coords(pixel, camera_K_inv, distortion_coeffs) # [u,v]
        # 归一化  o+t1d, 转到世界坐标系R(O+t1d)+t = t1*Rd + t
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(rotation_matrix, pc_d)
        ray_origins = translation_vector # TODO 可加速优化
        ray_directions = pw_d.flatten()

        # TODO 这里交了很多面片。需要优化到找到第一个面片就停止
        # 射线交面片：
        result = mesh_raycast.raycast(ray_origins, ray_directions, mesh=self.mesh)
        if len(result) == 0: # TODO 可优化
            result_point = np.array([0.,0,0]).reshape(3,1)
        else:
            first_result = min(result, key=lambda x: x['distance'])
            result_point= first_result['point']
        return result_point.tolist()

    def process(self, data: Package):
        data.location = self.get_point(data)
        
        