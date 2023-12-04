import json
import random
import tools.utils as utils
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--mesh_path', default='data/odm_textured_model_geo.obj', type=str, help='Path of map mesh')
    parser.add_argument('--num_points', default=500, type=int,
                        help='Required amount of simulated data')
    parser.add_argument('--gen_unity',  action='store_true',
                        help='Whether to generate unity data')
    parser.add_argument('--gen_sim',  action='store_true',
                        help='Whether to generate sim data')
    parser.add_argument('--unity_data', default='data/unit_data.json',
                        type=str, help='Path of unit_data')
    parser.add_argument('--sim_data', default='data/simulate_data.json',
                        type=str, help='Path of sim_data')
    args = parser.parse_args()

    # 仿真数据量
    NUM_PERSON = args.num_points
    NUM_CAR = args.num_points

    # 地图路径
    MESH_PATH = args.mesh_path

    # 相机参数
    # 默认25fps，每40ms拍摄一次。
    # 人类平均速度为2m/s,汽车20m/s。换算每帧之间0.08m，0.8m

    # 相机内参  [fx, fy, cx, cy]
    K = [355.72670241928597, 357.6787245904993,
         311.9712774887887, 253.00946170247045]

    # 相机位姿 [yaw, pitch, roll, x, y ,z]
    pose = [0., -30., 180, -10.89, 39.98, 150.]

    start_point_human = [pose[3]-8, pose[4]+30]  # 人类起始位置

    start_point_vehicle = [pose[3]-8, pose[4]+60]  # 车辆起始位置

    # 相机畸变 [k1,k2,k3,p1,p2]
    distortion = [0., 0., 0., 0., 0.]

    sim_camrea = utils.SimulationCamera(pose, K, distortion, MESH_PATH)

    unity_data = {
        "data": []
    }

    human_data = []
    car_data = []

    # 1.生成人类的真实世界坐标系位置，[u,v,w,h],预测位置 0.08m/帧
    for i in range(NUM_PERSON):
        # 找到真实点
        now_point = start_point_human[:]
        # idx = random.randint(0, 1)  # x,y 两个方向随机
        idx = 0
        now_point[idx] = now_point[idx] + i*0.08
        gt_point, bbox, pred = sim_camrea.generate_simulation(now_point, [
                                                              10, 30])
        human_data.append([gt_point, bbox])
        unity_data["data"].append({"id": 10, "point": gt_point})
        unity_data["data"].append({"id": 11, "point": pred})

    # 1.生成车辆的真实世界坐标系位置，[u,v,w,h],预测位置 0.8m/帧
    for i in range(NUM_CAR):
        # 找到真实点
        now_point = start_point_vehicle[:]
        idx = random.randint(0, 1)  # x,y 两个方向随机
        now_point[idx] = now_point[idx] + i*0.8
        gt_point, bbox, pred = sim_camrea.generate_simulation(now_point, [
                                                              40, 60])
        car_data.append([gt_point, bbox])
        if args.gen_unity:
            unity_data["data"].append({"id": 20, "point": gt_point})
            unity_data["data"].append({"id": 21, "point": pred})
        # print("gt_point", gt_point)
        # print("pred", pred)

    if args.gen_unity:
        with open(args.unity_data, 'w') as outfile:
            json_data = json.dumps(unity_data)
            outfile.write(json_data)

    start_timestamp = 1701482850000  # unix 时间戳 2023-12-02 10:07:30.000 ms 起始时间
    uav_id = 1
    camera_id = 1

    data = {
        "timestamp": None,
        "uav_id": uav_id,
        "camera_id": camera_id,
        "camera_params": {
            "pose": pose,  # [yaw,pitch,roll,x,y,z]
            "K": K,        # [fx,fy,cx,cy]
            "distortion": distortion,  # [k1,k2,p1,p2]
            "obj_img": None
        },
        "obj_cnt": None,
        "objs": [],
    }

    obj = {
        "tracker_id": 1,
        "cls_id": 1,
        "bbox": [
            39.49999532739537,
            237.00004132221807,
            1.0,
            3.0
        ],
        "loc": None
    }

    sim_data = {"data": []}

    # 生成json数据
    for i in range(len(human_data)):
        data["timestamp"] = start_timestamp + i*40 + random.randint(-10, 10)
        data["obj_cnt"] = 2

        # 人
        obj["tracker_id"] = 1
        obj["cls_id"] = 0
        obj["bbox"] = human_data[i][1]
        obj["loc"] = human_data[i][0]
        data["objs"].append(obj.copy())

        # 车
        obj["tracker_id"] = 2
        obj["cls_id"] = 1
        obj["bbox"] = car_data[i][1]
        obj["loc"] = car_data[i][0]
        data["objs"].append(obj.copy())

        sim_data["data"].append(data.copy())    # 深拷贝
    if args.gen_sim:
        with open(args.sim_data, 'w') as outfile:
            json_data = json.dumps(sim_data)
            outfile.write(json_data)
