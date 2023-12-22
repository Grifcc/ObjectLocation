from framework import Package
from framework import PreProcess
import copy


class TimeFilter(PreProcess):
    def __init__(self, time_slice, max_queue_length=None):
        super().__init__("TimeFilter", time_slice, max_queue_length)

    def process(self, data: list[Package]):
        return_data = []  # 需要返回的列表
        data_map = {}  # 用于存储数据的字典，按无人机id分类
        img_map  = {}

        for package in data:
            map_key = f"{package.uav_id}_{package.tracker_id}"
            if package.obj_img !=None:
                img_map[map_key] = copy.deepcopy(package.obj_img)
            if map_key not in data_map.keys():
                data_map[map_key] = package
            else:
                if data_map[map_key].time < package.time:
                    data_map[map_key] = package
                    
                     
        for k,v in data_map.items():
            if k in img_map.keys():
                v.obj_img = img_map[k]
            return_data.append(v)

        return return_data
