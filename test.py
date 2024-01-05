from modules import SpatialFilter
import json
from tools import ParseMsg, generate_n_colors
import time


parse = ParseMsg("data/map/JiuLongLake_v1223/offset.txt")

sp = SpatialFilter(1000, 4, 10)

log_path = "log\\mqtt_source\\20240104_13h34m19s.log"
with open(log_path, "r") as f:
    data = [json.loads(i) for i in f.readlines()]

packages = []
for mqtt_package in data:
    objs = parse.parse_msg(mqtt_package)
    packages.extend(objs[:])  # 添加到队列

packages = sorted(packages, key=lambda x: x.time)

process_data = []
for i in packages:
    if process_data == []:
        process_data.append(i)
    elif i.time - process_data[0].time > 1000:
        break
    else:
        process_data.append(i)

t1 = time.time()
x = sp.process(process_data)
t2 = time.time()
print("time: ", t2-t1)

print("total packages: ", len(packages))
