import json
import random
from tools.utils import PointType
import tools.utils as utils
import argparse
import numpy as np
import copy
from tqdm import tqdm

data_json = {
    "timestamp": None,
    "uav_id": None,
    "camera_id": None,
    "camera_params": {
        "pose": None,  # [yaw,pitch,roll,x,y,z]
        "K": None,        # [fx,fy,cx,cy]
        "distortion": None,  # [k1,k2,p1,p2]
    },
    "obj_cnt": None,
    "objs": [],
}

obj_json = {
    "uid": None, # for evaluation
    "tracker_id": None,
    "cls_id": None,
    "bbox": [],
    "loc": None,
    "obj_img": None
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--mesh_path', default='data/odm_textured_model_geo.obj', type=str, help='Path of map mesh')
    parser.add_argument('--duration', default=30, type=int,
                        help='Required amount of simulated data,unit: second')
    parser.add_argument('--fps', default=24, type=int,
                        help='Frame rate of camera')
    parser.add_argument('--speed', default=[2, 20], type=list,
                        help="The object's movement speed, default:person 2 m/s, car 20 m/s [2,20].")
    parser.add_argument('--shape', default=[640, 480], type=list,
                        help="simulated image size, default:[640,480]")
    parser.add_argument('--gen_unity',  action='store_true',
                        help='Whether to generate unity data')
    parser.add_argument('--gen_sim',  action='store_true',
                        help='Whether to generate sim data')
    parser.add_argument('--unity_data', default='data/unit_data.json',
                        type=str, help='Path of unit_data')
    parser.add_argument('--sim_data', default='data/simulate_data.json',
                        type=str, help='Path of sim_data')
    args = parser.parse_args()

    # 仿真帧率
    FRAME_RATE = args.fps
    # 仿真时间
    DURATION = args.duration
    # 仿真数据量
    NUM_FRAME = FRAME_RATE * DURATION

    # 仿真物体运动速度
    SPEED = args.speed

    # 图像大小
    SHAPE = args.shape

    # 地图路径
    MESH_PATH = args.mesh_path

    # 内参、畸变参数、外参
    K = [355.72670241928597, 357.6787245904993,
         311.9712774887887, 253.00946170247045]
    distortion = [0., 0., 0., 0., 0.]
    cam_start = [-210., 0., 100.]
    pose1 = [0., -30., 180, cam_start[0], cam_start[1], cam_start[2]]
    pose2 = [0., 30, 180, cam_start[0]+290., cam_start[1]-15., cam_start[2]]
    pose3 = [0., 0, 150, cam_start[0]+210., cam_start[1]+100., cam_start[2]]

    camera1 = utils.SimulationCamera(pose1, K, distortion, MESH_PATH, SHAPE)
    camera2 = utils.SimulationCamera(pose2, K, distortion, MESH_PATH, SHAPE)
    camera3 = utils.SimulationCamera(pose3, K, distortion, MESH_PATH, SHAPE)
    camera_list = [camera1, camera2, camera3]

    # 目标的起始xy位置
    person1 = [100., 0.]  # 右下角走 # # TODO 这个起始点很难选。容易观测不到 ，我这里只是试一下
    car2 = [-200., 80.]  # 右下角走
    person3 = [-200., 50.]  # 右上角走
    car4 = [-200., -120.]  # 右上角走
    person5 = [-200., -140.]  # 右上角走

    objs_start_points = [person1, car2, person3, car4, person5]
    objs_ids = [0, 1, 0, 1, 0]
    BBOX_SZIE = [[10, 20], [40, 20]]
    objs = []

    for idx, start_point in enumerate(objs_start_points):
        clsid = objs_ids[idx]
        objs.append(utils.SimulationObject(
            start_point, SPEED[clsid]/FRAME_RATE, BBOX_SZIE[clsid], clsid, 45, len(camera_list), max_age=2, uid=idx))

    unity_data = {
        "data": []
    }
    sim_data = {"data": []}

    start_timestamp = 1701482850000  # unix 时间戳 2023-12-02 10:07:30.000 ms 起始时间

    for i in tqdm(range(NUM_FRAME)):
        for uav_id, camera in enumerate(camera_list):
            sim_package = copy.deepcopy(data_json)
            sim_package["timestamp"] = start_timestamp + \
                i*40 + random.randint(-10, 10)
            sim_package["uav_id"] = uav_id
            sim_package["camera_id"] = uav_id
            sim_package["camera_params"]["pose"], sim_package["camera_params"][
                "K"], sim_package["camera_params"]["distortion"] = camera.get_params()

            for idx in range(len(objs)):

                if objs[idx].age[uav_id] > objs[idx].max_age:
                    # 超过最大age，表示跟丢，重置tracker_id,重置age
                    objs[idx].tracker_id[uav_id] = -1
                    objs[idx].reset_age(uav_id)

                sim_obj_data = copy.deepcopy(obj_json)
                status, data = camera.get_bbox_result(
                    objs[idx].next_point(), objs[idx].get_bbox_size())
                if status == PointType.ValidPoint:
                    if objs[idx].tracker_id[uav_id] == -1:
                        # 新目标, 赋值tracker_id
                        objs[idx].tracker_id[uav_id] = camera.get_max_id()
                        camera_list[uav_id].max_id += 1

                    sim_obj_data["tracker_id"] = objs[idx].tracker_id[uav_id]
                    sim_obj_data["cls_id"] = objs[idx].get_clsid()
                    sim_obj_data["bbox"] = data[0]
                    sim_obj_data["loc"] = data[1]
                    sim_obj_data["uid"] = objs[idx].uid

                    sim_package["objs"].append(sim_obj_data)
                    objs[idx].reset_age(uav_id)  # 重置age
                else:
                    objs[idx].age[uav_id] += 1

            sim_package["obj_cnt"] = len(sim_package["objs"])
            if not sim_package["obj_cnt"] == 0:
                sim_data["data"].append(sim_package)

    # if args.gen_unity:
    #     with open(args.unity_data, 'w') as outfile:
    #         json_data = json.dumps(unity_data)
    #         outfile.write(json_data)

    if args.gen_sim:
        with open(args.sim_data, 'w') as outfile:
            json_data = json.dumps(sim_data)
            outfile.write(json_data)
