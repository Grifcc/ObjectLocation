import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils


# 图像长宽
W = 640
H = 480

# 相机内参  [fx, fy, cx, cy]
K = [355.72670241928597, 357.6787245904993,
         311.9712774887887, 253.00946170247045]
K, K_inv = utils.set_K(K)
distortion = [0., 0., 0., 0., 0.]
# 相机位姿
pose1 = [180., -30, 0, -10., -15, 6]
pose2 = [180., 30, 0, 10., -15, 6]
pose3 = [150., 0, 0, 0., -25, 6]
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

# 读取OBJ文件
# obj_file_path = 'E:\\code\\ObjectLocation\\data\\Desert.obj'

def drawPoint(ax, t1, points, color):
    # 绘制四个点
    ax.scatter(t1[0], t1[1], t1[2], c=color, marker='o')
    # 依次连接四个点
    for i in range(len(points)):
        ax.plot([t1[0], points[i][0]], [t1[1], points[i][1]], [t1[2], points[i][2]], c=color, linestyle='--')
    # 依次连接四个点
    for i in range(len(points) - 1):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], [points[i][2], points[i + 1][2]], c=color)
    ax.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]],  [points[3][2], points[0][2]], c=color)

print(t1)
t1 = t1.flatten()
t2 = t2.flatten()
t3 = t3.flatten()
print(points1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def drawPoint(ax, t1, points, color):
    # 绘制四个点
    ax.scatter(t1[0], t1[1], t1[2], c=color, marker='o')
    # 依次连接四个点
    for i in range(len(points)):
        ax.plot([t1[0], points[i][0]], [t1[1], points[i][1]], [t1[2], points[i][2]], c=color, linestyle='--')
    # 依次连接四个点
    for i in range(len(points) - 1):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], [points[i][2], points[i + 1][2]], c=color)
    ax.plot([points[3][0], points[0][0]], [points[3][1], points[0][1]],  [points[3][2], points[0][2]], c=color)

drawPoint(ax, t1, points1, color='r')
drawPoint(ax, t2, points2, color='g')
drawPoint(ax, t3, points3, color='b')

def read_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex.split('/')[0]) for vertex in line.strip().split()[1:]]
                faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces, dtype = object)
    # faces = np.array(faces)
    vertices[:,1] = vertices[:,1] - min(vertices[:,1])

    return vertices, faces
  
def visualize_obj(ax, vertices, faces):
    # 绘制顶点
    #ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r', marker='o', s = 0.25)

    # 绘制面
    for face in faces:
        face_vertices = vertices[np.array(face) - 1]
        face_vertices = np.vstack([face_vertices, face_vertices[0]])  # 将面闭合
        ax.plot(face_vertices[:, 0], face_vertices[:, 2], face_vertices[:, 1], c='b', linewidth=0.1)


# 可视化OBJ文件
# vertices, faces = read_obj(obj_file_path)
# visualize_obj(ax, vertices, faces)

def track(ax, points):
    for i in range(1):
        ax.scatter(points[0][0], points[1][0], points[2][0], c='b', marker='o')
real_point = np.array([-6.366050, -30.999302 ,-0.982937 ]).reshape(3,1)
track(ax, real_point)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()


