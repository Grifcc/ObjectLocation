import json
import math


 # 仿真帧率
FRAME_RATE = 24
# 仿真时间
DURATION = 30


def get_angle(p1, p2):
    if p2[0]-p1[0] != 0:
        slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    else:
        if p2[1]-p1[1] > 0:
            return 90
        else:
            return -90
    return  math.degrees(math.atan(slope))

def get_distance(p1, p2):
    return math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)



with open("data/objects.json", "r") as f:
    ori_data = json.load(f)


for i in range(len(ori_data["objects"])):
    start = ori_data["objects"][i]["start_point"]
    end = ori_data["objects"][i]["end_point"]
    ori_data["objects"][i]["angle"] = get_angle(start, end)
    ori_data["objects"][i]["distance"] = get_distance(start, end)
    ori_data["objects"][i]["speed"] = ori_data["objects"][i]["distance"] / DURATION
    if ori_data["objects"][i]["speed"] > 0.3:
        ori_data["objects"][i]["cls_id"] = 1 # fast 车
    else:
        ori_data["objects"][i]["cls_id"] = 0 # slow 人
    ori_data["objects"][i]["uid"] = i

ori_data["fps"] = FRAME_RATE
ori_data["duration"] = DURATION

with open("data/objects.json", "w") as f:
    json.dump(ori_data, f, indent=4)
