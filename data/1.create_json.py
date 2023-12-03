import json
import os
# 单相机多时刻2目标跟踪

def create_json_file(timestamp, uav_id, camera_id, pose, K, distortion, tracked_data, i):
    data = {
        "timestamp": timestamp,
        "uav_id": uav_id,
        "camera_id": camera_id,
        "camera_params": {
            "pose": pose,  # [yaw,pitch,roll,x,y,z]
            "K": K,        # [fx,fy,cx,cy]
            "distortion": distortion  # [k1,k2,p1,p2]
        },
        "obj_cnt": len(tracked_data),
        "objs": [],
        "obj_img": None
    }

    for tracked_item in tracked_data:
        obj = {
            "tracker_id": tracked_item[0],
            "cls_id": tracked_item[1],  # 0车1人
            "bbox": tracked_item[2],    # [x,y,w,h]
            "loc": None      # (WGS84）
        }
        data["objs"].append(obj)

    # Writing data to JSON file
    folder_path = 'jsons'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open('jsons/time'+str(i)+'.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

# 默认25fps，每40ms拍摄一次。
# 人类平均速度为2m/s,汽车20m/s。换算每帧之间0.08m，0.8m
import trimesh
import mesh_raycast
import numpy as np
import time

def set_K(cam_K):
    fx,fy,cx,cy = cam_K
    # 构建内参矩阵
    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
    K_inv = np.linalg.inv(K)
    return K, K_inv
    
def set_distortion_coeffs( distortion_param):
    k1, k2, k3, p1, p2 = distortion_param
    return np.array([k1, k2, k3, p1, p2])

def set_rotation_matrix(t1, t2, t3): # (roll, pitch, yaw)
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
    
def set_translation_vector(t1, t2, t3):
    return np.array([[t1], [t2], [t3]]).reshape(3,1)

def set_camera_pose(camera_pose):
    rotation_matrix = set_rotation_matrix(t1=camera_pose[2],t2=camera_pose[1],t3=camera_pose[0])
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


# 相机参数
K = [355.72670241928597, 357.6787245904993, 311.9712774887887, 253.00946170247045] 
distortion = [0., 0., 0., 0., 0.]
pose = [0., -30., 180, -18., -234., 150.]
camera_K, camera_K_inv = set_K(K)
rotation_matrix, translation_vector = set_camera_pose(pose)
camera_rotation_inv =np.linalg.inv(rotation_matrix)
distortion_coeffs = set_distortion_coeffs(distortion)
# 读取mesh
path = 'odm_textured_model_geo.obj'
mesh = trimesh.load_mesh(path)
all_meshes = [geom for geom in mesh.geometry.values()]
# 使用 concatenate 函数将多个 mesh 合并为一个
combined_mesh = trimesh.util.concatenate(all_meshes)
vertices = combined_mesh.vertices
faces = combined_mesh.faces
triangles = vertices[faces]
triangles = np.array(triangles, dtype='f4')  # 一定要有这一行，不然会有错。
mesh = triangles
# 1.生成人类的真实世界坐标系位置，[u,v,w,h],预测位置 0.08m/帧
now_port = [-26.494522094726562, -232.09425354003906]
obj_1_real = []
obj_1_pixel = []
obj_1_pred = []
for i in range(20):
    # 找到真实点
    now_port[0] = now_port[0] + i*0.08
    ray_origins = np.array([now_port[0], now_port[1], 200]).reshape(3,1)
    ray_directions = [0,0,-1.]
    result = mesh_raycast.raycast(ray_origins, ray_directions, mesh)
    if len(result) == 0: # TODO 可优化
        result_point = np.array([0.,0,0]).reshape(3,1)
    else:
        first_result = min(result, key=lambda x: x['distance'])
        result_point= first_result['point']
    result_point = list(result_point)
    obj_1_real.append(result_point)
    print(result_point)
    # 找到pixel
    result_point = np.array(result_point).reshape(3,1)
    p_c1 = camera_rotation_inv@(result_point-translation_vector)
    p_cd = p_c1/p_c1[2]
    pixel = camera_K@p_cd
    obj_1_pixel.append([pixel[0][0]-0.5, pixel[1][0]-3.,1.,3.]) # TODO 注意这里人类的话1*3
    print([pixel[0][0]-0.5, pixel[1][0]-3.,1.,3.])
    # 找到预测点
    pixel = [pixel[0][0], pixel[1][0]]
    p_c1 = undistort_pixel_coords(pixel, camera_K_inv, distortion_coeffs)
    pc_d = p_c1 / np.linalg.norm(p_c1)
    pw_d = np.dot(rotation_matrix, pc_d)
    ray_origins = translation_vector
    ray_directions = pw_d.flatten()
    result = mesh_raycast.raycast(ray_origins, ray_directions, mesh)
    if len(result) == 0: # TODO 可优化
        result_point = np.array([0.,0,0]).reshape(3,1)
    else:
        first_result = min(result, key=lambda x: x['distance'])
        result_point= first_result['point']
    obj_1_pred.append(list(result_point))
    print((list(result_point)))

# 2.生成车的真实世界坐标系位置，[u,v,w,h],预测位置 0.8m/帧
now_port = [-26.494522094726562, -200.09425354003906]
obj_2_real = []
obj_2_pixel = []
obj_2_pred = []
for i in range(20):
    # 找到真实点
    now_port[0] = now_port[0] + i*0.8 # 0.8m/帧
    ray_origins = np.array([now_port[0], now_port[1], 200]).reshape(3,1)
    ray_directions = [0,0,-1.]
    result = mesh_raycast.raycast(ray_origins, ray_directions, mesh)
    if len(result) == 0: # TODO 可优化
        result_point = np.array([0.,0,0]).reshape(3,1)
    else:
        first_result = min(result, key=lambda x: x['distance'])
        result_point= first_result['point']
    result_point = list(result_point)
    obj_2_real.append(result_point)
    print(result_point)
    # 找到pixel
    result_point = np.array(result_point).reshape(3,1)
    p_c1 = camera_rotation_inv@(result_point-translation_vector)
    p_cd = p_c1/p_c1[2]
    pixel = camera_K@p_cd
    obj_2_pixel.append([pixel[0][0]-2., pixel[1][0]-3.,4.,6.]) # TODO 注意这里人类的话4*6
    print([pixel[0][0]-2, pixel[1][0]-3.,4.,6.])
    # 找到预测点
    pixel = [pixel[0][0], pixel[1][0]]
    p_c1 = undistort_pixel_coords(pixel, camera_K_inv, distortion_coeffs)
    pc_d = p_c1 / np.linalg.norm(p_c1)
    pw_d = np.dot(rotation_matrix, pc_d)
    ray_origins = translation_vector
    ray_directions = pw_d.flatten()
    result = mesh_raycast.raycast(ray_origins, ray_directions, mesh)
    if len(result) == 0: # TODO 可优化
        result_point = np.array([0.,0,0]).reshape(3,1)
    else:
        first_result = min(result, key=lambda x: x['distance'])
        result_point= first_result['point']
    obj_2_pred.append(list(result_point))
    print((list(result_point)))

print(len(obj_1_real), len(obj_1_pixel), len(obj_1_pred))
print(len(obj_2_real), len(obj_2_pixel), len(obj_2_pred))





timestamp = 124567824564654
uav_id = 1
camera_id = 1
# [yaw, pitch, roll, x, y ,z]
pose = [0., -30., 180, -18., -234., 150.]
# [fx, fy, cx, cy]
K = [355.72670241928597, 357.6787245904993, 311.9712774887887, 253.00946170247045] 
# [k1,k2,k3,p1,p2]
distortion = [0., 0., 0., 0., 0.]
# [tracker_id, cls_id, bbox, loc]
traked_num = 2
tracked_data = [[1,0,[30,30,3,4],[]],[2,1,[20,30,1,2],[]]]
tracked_data = []
time = len(obj_1_real)
for i in range(time):
    timestamp = timestamp + i*40 # 40ms一帧
    tracked_data.append([1,1,obj_1_pixel[i]]) # 人
    tracked_data.append([2,0,obj_2_pixel[i]]) # 车
    
    create_json_file(timestamp, uav_id, camera_id, pose, K, distortion, tracked_data, i)