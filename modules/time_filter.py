from framework import Package
from framework import PreProcess
import copy
class TimeFilter(PreProcess):
    def __init__(self, time_slice):
        super().__init__("TimeFilter", time_slice)
        
    def process(self, data: list[Package]): 
        return_data = []  #需要返回的列表
        data_map = {} #用于存储数据的字典，按无人机id分类
        for package in data:
            map_key = f"{package.uav_id}_{package.tracker_id}"
            if map_key  not in data_map.keys():
                data_map[map_key] = package
            else:
                if data_map[map_key].time < package.time:
                    data_map[map_key] = package

        for package in data_map.values():
            return_data.append(package)

        data =copy.deepcopy(return_data)
        