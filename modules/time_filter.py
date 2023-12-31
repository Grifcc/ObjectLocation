from framework import Package
from framework import PreProcess

class TimeFilter(PreProcess):
    def __init__(self, time_slice, max_queue_length=None):
        super().__init__("TimeFilter", time_slice, max_queue_length)

    def process(self, data: list[Package]):
        return_data = []  # 需要返回的列表
        data_map = {}  # 用于存储数据的字典，按无人机id分类

        for package in reversed(data):
            map_key = f"{package.uav_id}_{package.tracker_id}"
            if map_key not in data_map.keys():
                data_map[map_key] = package
            else:
                if data_map[map_key].time < package.time:
                    if  data_map[map_key].obj_img != None:
                        package.obj_img =data_map[map_key].obj_img
                    data_map[map_key] = package
                
        for package in data_map.values():
            return_data.append(package)

        return return_data
