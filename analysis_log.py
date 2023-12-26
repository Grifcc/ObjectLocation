import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from tools import UWConvert, generate_n_colors

log_path = "./data/20231222_21h35m35s_TH7923461373.log"
with open(log_path, "r") as f:
    data = [json.loads(i) for i in f.readlines()]

data = [i for i in data if i["obj_cnt"] != 0]


convert = UWConvert("data\map\JiuLongLake_v1223\offset.txt")

points = {}
for objs in data:
    for obj in objs["objs"]:
        if obj["track_id"] not in points.keys():
            points[obj["track_id"]] = []
        else:
            points[obj["track_id"]].append(convert.W2U(
                [obj["pos"]["latitude"], obj["pos"]["longitude"], obj["pos"]["altitude"]]))

colors = generate_n_colors(len(points.keys()))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

color_idx = 0
for k, v in points.items():
    for i in v:
        ax.scatter(i[0], i[1], i[2], c=colors[color_idx])
    color_idx += 1

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
