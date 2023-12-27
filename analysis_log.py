from modules import EstiPosition
from tools import ParseMsg, generate_n_colors
import json
import matplotlib.pyplot as plt
from tracker import Sort
import numpy as np


RELOACTION = True
REID = True

log_path = "log/log.log"

parse = ParseMsg("data\map\JiuLongLake_v1223\offset.txt")


esti = EstiPosition(mesh_path="data\map\JiuLongLake_v1223\mesh.obj",
                    enable=False)

tracker = Sort(10, 4, 3)  # max age, min hits, dis threshold
with open(log_path, "r") as f:
    data = [json.loads(i) for i in f.readlines()]

data = sorted(data, key=lambda x: x["time"])
print("total packages: ", len(data))
print("total time: ", data[-1]["time"] - data[0]["time"])


# 画时间差分
y = []
for i in range(1, len(data)):
    y.append(data[i]["time"] - data[i-1]["time"])

plt.plot(list(range(len(y))), y)
plt.savefig("delta_time.png")
plt.close()


packages = []
for mqtt_package in data:
    objs = parse.parse_msg(mqtt_package)
    points = []
    cls_cnt = 0
    for idx, obj in enumerate(objs):
        objs[idx].location = esti.get_point_form_uav_object_point(
            obj) if RELOACTION else obj.uav_utm[:]
        points.append([*objs[idx].location, obj.class_id])

    if REID:
        points = np.array(points).reshape(-1, 4)
        ret = tracker.update(points)
        for i, t in enumerate(ret):
            objs[i].tracker_id = t[4]

    packages.extend(objs)  # 添加到队列

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
        ax.scatter(tracklets[id][idx].location[0], tracklets[id]
                   [idx].location[1], tracklets[id][idx].location[2], c=colors[id])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
