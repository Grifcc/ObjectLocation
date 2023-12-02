import json
import random

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
            "loc": tracked_item[3]      # (WGS84）
        }
        data["objs"].append(obj)

    # Writing data to JSON file
    with open('jsons/'+str(timestamp)+'.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

start_timestamp = 1701482850000  #unix 时间戳 2023-12-02 10:07:30.000 ms  起始时间
uav_id = 1
camera_id = 1
# [yaw, pitch, roll, x, y ,z]
pose = [0., 0., 150., 0., -25., 6.]   
# [fx, fy, cx, cy]
K = [355.72670241928597, 357.6787245904993, 311.9712774887887, 253.00946170247045] 
# [k1,k2,p1,p2]
distortion = [-0.16511311963465314, 0.06499598603806166, 0.0033307309789138034, 0.007443839439139966]
# [tracker_id, cls_id, bbox, loc]
traked_num = 2
tracked_data = [[1,0,[30,30,3,4],[]],[2,1,[20,30,1,2],[]]]
package_num = 1000  #需要模拟的数据包数量
for i in range(package_num):
    timestamp = start_timestamp + i*1000 + random.randint(-30, 30)  #  有随机误差的时间戳  +-30ms
    for j in range(len(tracked_data)):
        # 像素位置更新
        if tracked_data[j][1] == 0: # 车，每次移动4像素
            tracked_data[j][2][0] =  tracked_data[j][2][0] + 4
            tracked_data[j][2][1] =  tracked_data[j][2][1] + 4
        elif tracked_data[j][1] == 1: # 人，每次移动1像素
            tracked_data[j][2][0] =  tracked_data[j][2][0] + 2
            tracked_data[j][2][1] =  tracked_data[j][2][1] + 2     
    create_json_file(timestamp, uav_id, camera_id, pose, K, distortion, tracked_data, i)
