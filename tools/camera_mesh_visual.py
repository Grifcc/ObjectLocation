import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils

# 画一个单点
def track(ax, points):
    for i in range(1):
        ax.scatter(points[0][0], points[1][0], points[2][0], c='b', marker='o')

# 图像分辨率
W = 640
H = 480

# 相机内参  [fx, fy, cx, cy]
K = [355.72670241928597, 357.6787245904993, 311.9712774887887, 253.00946170247045]
K, K_inv = utils.set_K(K)

# 相机位姿 [yaw, pitch, roll, x, y, z] rgb
cam_start = [-210., 0., 100.]
pose1 = [0., -30., 180, cam_start[0], cam_start[1], cam_start[2]]
pose2 = [0., 30, 180, cam_start[0]+290., cam_start[1]-15., cam_start[2]]
pose3 = [0., 0, 150, cam_start[0]+210., cam_start[1]+100., cam_start[2]]

# pose1 = [0., -30., 180, -210.89, 39.98, 70.]
# pose2 = [0., 30, 180, 210., -15, 150.]
# pose3 = [0., 0, 150, 0., 175, 150.]
R1, t1 = utils.set_camera_pose(pose1)
R2, t2 = utils.set_camera_pose(pose2)
R3, t3 = utils.set_camera_pose(pose3)

# 获取相机的四个视角锥
rays_o1, rays_d1 = utils.get_rays_corners(H, W, K, R1, t1)
rays_o2, rays_d2 = utils.get_rays_corners(H, W, K, R2, t2)
rays_o3, rays_d3 = utils.get_rays_corners(H, W, K, R3, t3)

# 获取相机的视野
points1 = utils.compute_xy_coordinate(rays_o1, rays_d1)
points2 = utils.compute_xy_coordinate(rays_o2, rays_d2)
points3 = utils.compute_xy_coordinate(rays_o3, rays_d3)

points1 = np.array(points1)+np.array([0, 0, 80]).reshape(1, 3)
faces1 = np.array([
    [0, 1, 3],
    [1, 2, 3],
])
# 创建一个新的 Trimesh 对象
new_mesh1 = trimesh.Trimesh(vertices=points1, faces=faces1)

points2 = np.array(points2)+np.array([0, 0, 80]).reshape(1, 3)
faces2 = np.array([
    [0, 1, 3],
    [1, 2, 3],
])
# 创建一个新的 Trimesh 对象
new_mesh2 = trimesh.Trimesh(vertices=points2, faces=faces2)

points3 = np.array(points3)+np.array([0, 0, 80]).reshape(1, 3)
faces3 = np.array([
    [0, 1, 3],
    [1, 2, 3],
])
# 创建一个新的 Trimesh 对象
new_mesh3 = trimesh.Trimesh(vertices=points3, faces=faces3)



# 加载已有的 OBJ 文件
obj_file_path = 'D:\\code\\location\\ObjectLocation\\data\\odm_textured_model_geo.obj'
existing_mesh = trimesh.load_mesh(obj_file_path)

# 创建一个新的 Scene
combined_scene = trimesh.Scene()

# 将两个网格添加到 Scene 中
combined_scene.add_geometry({
    'mesh1': existing_mesh,
    'mesh2': new_mesh1,
    'mesh3': new_mesh2,
    'mesh4': new_mesh3
})

# 显示 Scene
combined_scene.show()

