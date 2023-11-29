from framework import Package
from framework import Location
import trimesh
import numpy as np
import mesh_raycast
import time
import numpy as np
from scipy.spatial.transform import Rotation

class EstiPosition(Location):
    def __init__(self,cam_K=None, mesh_path=None):
        super().__init__("EstiPosition",)
        self.pixel = None # 像素坐标
        self.point = None  # 三维坐标
        self.rotation_matrix = None # 相机旋转矩阵
        self.rotation_matrix_inv = None # 相机旋转矩阵的逆
        self.translation_vector = None # 相机平移向量 
        self.camera_K_inv = None  # 相机内参的逆
        self.camera_K = self.set_K(cam_K)  # 相机内参
        self.mesh = self.set_Mesh(mesh_path)  # 网格

    def set_K(self, cam_K):
        fx,fy,cx,cy = cam_K
        # 构建内参矩阵
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        self.camera_K_inv = np.linalg.inv(K)
        return K
    
    def set_Mesh(self, path):
        mesh = trimesh.load_mesh(path) #load or load_mesh
        if (type(mesh) == trimesh.base.Trimesh):
            # 移动到中心
            mesh=mesh_centerize(mesh)
   
        # if (type(mesh) == trimesh.scene.scene.Scene):
        #     print("scene")
        #     print(list(mesh.geometry.values())[0])
        # elif (type(mesh) == trimesh.base.Trimesh):
        #     print("single object")
        #     # 移动到中心
        #     mesh=mesh_centerize(mesh)
        #     print(np.min(mesh.vertices,0))
        #     print(np.max(mesh.vertices, 0))
        return mesh
    
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
        self.rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        return rotation_matrix
    
    def set_camera_pose(self, camera_pose):
        self.rotation_matrix = self.set_rotation_matrix(t1=camera_pose[2],t2=camera_pose[1],t3=camera_pose[0])
        self.translation_vector = np.array([[camera_pose[3]], [camera_pose[4]], [camera_pose[5]]]).reshape(3,1)


    def set_pixel(self, cls_id, boundingbox):
        float_bbox = tuple(float(val) for val in boundingbox)
        matrix = np.array([[1.], [1.], [1.]])
        if(cls_id): # 1:车 (cx,cy)+(w/2. +h/2.)
            matrix[0] = float_bbox[0] + float_bbox[3]/2.
            matrix[1] = float_bbox[1] + float_bbox[2]/2.
        else: # 0:人 (cx,cy)+(w/2. +h)
            matrix[0] = float_bbox[0] + float_bbox[3]/2.
            matrix[1] = float_bbox[1] + float_bbox[2]
        self.pixel = matrix.reshape(3,1)

    def get_point(self):
        p_c1 = np.dot(self.camera_K_inv, self.pixel)
        # 归一化  o+t1d, 转到世界坐标系R(O+t1d)+t = t1*Rd + t
        pc_d = p_c1 / np.linalg.norm(p_c1)
        pw_d = np.dot(self.rotation_matrix, pc_d)
        ray_origins = self.translation_vector
        ray_directions = pw_d.flatten()

        real_point = np.array([-9.970698356628418, 11.595516204833984, 5.487466812133789]).reshape(3,1) 
        pred_point = []
        # 射线交面片：
        triangles = list(self.mesh.geometry.values())[0].vertices[list(self.mesh.geometry.values())[0].faces] # TODO 这两句可优化
        triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
        result = mesh_raycast.raycast(ray_origins, ray_directions, mesh=triangles)
        if len(result) == 0: # TODO 可优化
            self.point = np.array([0.,0,0]).reshape(3,1)
        else:
            first_result = min(result, key=lambda x: x['distance'])
            self.point = first_result['point']
            pred_point.append(first_result['point'])

    def process(self, data: Package):
        pass
        