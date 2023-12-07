import utils
import numpy as np

# 内参、畸变参数、外参
K = [355.72670241928597, 357.6787245904993,
    311.9712774887887, 253.00946170247045]
distortion = [0., 0., 0., 0., 0.]
cam_start = [-210., 0., 100.]
pose1 = [0., -30., 180, cam_start[0], cam_start[1], cam_start[2]]
pose2 = [0., 30, 180, cam_start[0]+290., cam_start[1]-15., cam_start[2]]
pose3 = [0., 0, 150, cam_start[0]+210., cam_start[1]+100., cam_start[2]]
poses = [pose1, pose2, pose3]

MESH_PATH = "/root/workspace/code/ex1/aaa_location/ObjectLocation/data/odm_textured_model_geo.obj"
sim_camrea = utils.SimulationCamera(poses, K, distortion, MESH_PATH)
timestamp = 1701482850000 

# 目标的xy位置 
person1 = [100., 0.] # 右下角走 # # TODO 这个起始点很难选。容易观测不到 ，我这里只是试一下
car2 = [-200., 80.] # 右下角走
person3 = [-200., 50.] # 右上角走
car4 = [-200., -120.] # 右上角走
person5 = [-200., -140.] # 右上角走


# 默认25fps，每40ms拍摄一次。
# 人类平均速度为2m/s,汽车20m/s。换算每帧之间0.08m，0.8m 45度换算为xy轴速度为0.0566 0.566

track_id = [0,0,0] # 每台相机对这个目标的track_id

def track(person1, class_id, number, sim_camera):
    sim_camera.track_reset()
    for i in range(number):
        now_timestamp = timestamp + i*40
        now_xy = [person1[0]-0.0566*i, person1[1]] # TODO 人要判断
        cls_id = class_id # TODO 
        obj_size = [3, 6] # TODO
        sim_camera.start_set(now_timestamp, now_xy, cls_id, obj_size)
        if(not sim_camera.get_real_z()): # 地图没有，直接退出
            break
        else:
            sim_camera.get_pixel_result(track_id)
track(person1, 1, 200, sim_camrea)