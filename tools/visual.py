import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# 图像长宽
W = 640
H = 480

# 定义相机内参
fx = 355  # fx 定义相机内参 设定焦距和主点坐标(图像分辨率为640*480)
fy = 355  # fy
cx = 320  # cx
cy = 240  # cy

# 构建相机内参矩阵
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 读取OBJ文件
# obj_file_path = 'E:\\code\\ObjectLocation\\data\\Desert.obj'


 # 设置相机位姿   
def pose_camera(theta1, theta2, theta3, x, y, z): # (roll, pitch, yaw)

    # 角度转弧度
    theta1 = np.radians(theta1) # 绕X轴旋转
    theta2 = np.radians(theta2)  # 绕Y轴旋转
    theta3 = np.radians(theta3)  # 绕Z轴旋转

    # 相机在原坐标系中的位置
    t = np.array([x, y, z], dtype=float)

    # 构建绕X、Y、Z轴旋转的矩阵
    R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta1), -np.sin(theta1)],
            [0, np.sin(theta1), np.cos(theta1)]])

    R_y = np.array([
            [np.cos(theta2), 0, np.sin(theta2)],
            [0, 1, 0],
            [-np.sin(theta2), 0, np.cos(theta2)]])

    R_z = np.array([
            [np.cos(theta3), -np.sin(theta3), 0],
            [np.sin(theta3), np.cos(theta3), 0],
            [0, 0, 1]])

    # 得到总的旋转矩阵
    R = R_z @ R_y @ R_x
    return R, t.reshape(3, 1)

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



# 设置三个相机的位姿
# 六个参数为：绕x,y,z轴的旋转角度，以及相机位置(x,y,z)
R1, t1 = pose_camera(180., -30, 0, -10., -15, 6)
R2, t2 = pose_camera(180., 30, 0, 10., -15, 6)
R3, t3 = pose_camera(150., 0, 0, 0., -25, 6)

# 获取相机的四个视角锥
rays_o1, rays_d1 = get_rays_corners(H, W, K, R1, t1)
rays_o2, rays_d2 = get_rays_corners(H, W, K, R2, t2)
rays_o3, rays_d3 = get_rays_corners(H, W, K, R3, t3)

# 获取相机的视野
points1 = compute_xy_coordinate(rays_o1, rays_d1)
points2 = compute_xy_coordinate(rays_o2, rays_d2)
points3 = compute_xy_coordinate(rays_o3, rays_d3)

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


