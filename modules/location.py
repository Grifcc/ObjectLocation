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
        if (type(mesh) == trimesh.base.Trimesh):
            # 移动到中心
            mesh=mesh_centerize(mesh)

        triangles = list(mesh.geometry.values())[0].vertices[list(mesh.geometry.values())[0].faces] # TODO 这两句可优化
        triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
   
        # if (type(mesh) == trimesh.scene.scene.Scene):
        #     print("scene")
        #     print(list(mesh.geometry.values())[0])
        # elif (type(mesh) == trimesh.base.Trimesh):
        #     print("single object")
        #     # 移动到中心
        #     mesh=mesh_centerize(mesh)
        #     print(np.min(mesh.vertices,0))
        #     print(np.max(mesh.vertices, 0))
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
        k1, k2, p1, p2 = distortion_param
        return np.array([k1, k2, p1, p2])

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

    def undistort_pixel_coords(pixel_coords, camera_K, distortion_coeffs):
        # 使用OpenCV进行畸变矫正
        undistorted_coords = cv2.undistortPoints(pixel_coords, camera_K, distortion_coeffs,P=camera_K)
        # 返回矫正后的像素坐标 [u,v]
        return undistorted_coords.squeeze()

    def get_point(self, data: Package):
        camera_K, camera_K_inv = self.set_K(cam_K=data.camera_K)
        distortion_coeffs = self.set_distortion_coeffs(data.camera_distortion)
        rotation_matrix, translation_vector = self.set_camera_pose(camera_pose=data.camera_pose)

        # 获取像素并去畸变
        pixel = data.get_center_point() # [float, float]
        pixel = np.array([pixel])
        pixel = self.undistort_pixel_coords(pixel, camera_K, distortion_coeffs) # [u,v]
        pixel = np.array([[pixel[0]], [pixel[1]], [1.]]).reshape(3,1)

        p_c1 = np.dot(camera_K_inv, pixel)
        # 归一化  o+t1d, 转到世界坐标系R(O+t1d)+t = t1*Rd + t
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(rotation_matrix, pc_d)
        ray_origins = translation_vector
        ray_directions = pw_d.flatten()

        # TODO 这里交了很多面片。需要优化到找到第一个面片就停止
        result_point = None
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
        
        