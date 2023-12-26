from framework import Package
from framework import Location
import mesh_raycast
from tools import get_ray, read_mesh, set_K, set_distortion_coeffs, set_camera_pose, UWConvert
import numpy as np

class EstiPosition(Location):
    # enable  true:根据相机位姿定位目标点 false:根据道通uav_obj_pose重定位目标点
    def __init__(self, mesh_path=None, default_height=60, order="rzyx", enable=True, max_queue_length=None):
        super().__init__("EstiPosition", max_queue_length)
        self.mesh = read_mesh(mesh_path)  # mesh地图
        self.order = order
        self.default_height = default_height
        self.enable = enable

    def get_point(self, data: Package):
        _, K_inv = set_K(data.camera_K)
        D = set_distortion_coeffs(data.camera_distortion)

        # TODO: 可能R逆 是
        R, t, _ = set_camera_pose(data.camera_pose, order=self.order)
        ray = -get_ray(data.get_center_point(), K_inv, D, R)

        result = mesh_raycast.raycast(
            t.flatten(), ray, self.mesh)
        if len(result) == 0:  # TODO
            l = (self.default_height - t) / -ray[2]
            # 计算交点坐标
            inter_point = t - l * ray.reshape(3, 1)
            return inter_point.flatten().tolist()
        else:
            return min(result, key=lambda x: x['distance'])[
                'point']
        
    def get_point_form_uav_object_point(self, data: Package): 
        p_camera = np.array(data.camera_pose[3:]).reshape(3,1)
        p_obj = np.array([data.uav_utm]).reshape(3,1)
        ray = p_obj - p_camera
        ray = ray / np.linalg.norm(ray)
        result = mesh_raycast.raycast(
            p_camera.flatten(), ray, self.mesh)
        if len(result) == 0:  # TODO
            l = (self.default_height - t) / -ray[2]
            # 计算交点坐标
            inter_point = t - l * ray.reshape(3, 1)
            return inter_point.flatten().tolist()
        else:
            return min(result, key=lambda x: x['distance'])[
                'point']


    def process(self, data: Package):
        data.location = self.get_point(
            data) if self.enable else self.get_point_form_uav_object_point(data)
