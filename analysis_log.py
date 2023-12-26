from modules import EstiPosition
from tools import ParseMsg, generate_n_colors
import json
import matplotlib.pyplot as plt
from tracker import Sort
import numpy as np


RELOACTION = False
REID = True

log_path = "log_sqr\\20231226_16h13m59s_80.log"
parse = ParseMsg("data\map\JiuLongLake_v1223\offset.txt")


esti = EstiPosition(mesh_path="data\map\JiuLongLake_v1223\mesh.obj",
                    enable=False)

tracker = Sort(15, 3, 3)  # max age, min hits, dis threshold
with open(log_path, "r") as f:
    data = [json.loads(i) for i in f.readlines()]
print("total packages: ", len(data))

packages = []
for i in data:
    packages.extend(parse.parse_msg(i))

packages = sorted(packages, key=lambda x: x.time)  # 时间排序
print("total time: ", packages[-1].time - packages[0].time)

if REID:
    pass

tracklets = {}
for i in packages:
    if i.tracker_id not in tracklets:
        tracklets[i.tracker_id] = []
    tracklets[i.tracker_id].append(i)
print("total tracklets: ", len(tracklets.keys()))

colors = dict(zip(list(tracklets.keys()),
              generate_n_colors(len(tracklets.keys()))))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for id, lets in tracklets.items():
    for idx, val in enumerate(lets):
        tracklets[id][idx].location = esti.get_point_form_uav_object_point(
            val) if RELOACTION   else val.uav_utm[:]
        ax.scatter(tracklets[id][idx].location[0], tracklets[id]
                   [idx].location[1], tracklets[id][idx].location[2], c=colors[id])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
