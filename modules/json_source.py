from framework import Package
from framework import Source
import json


class JsonSource(Source):
    def __init__(self, json_file):
        super().__init__("json_source")
        self.json_file = json_file
        self.data = self.read_json(self.json_file)

    def read_json(self, json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
        return json_data["data"]

    def process(self,packages: list[Package]):
        if len(self.data) == 0:
            return False
        objs = self.data.pop(0)
        bbox = Package()
        
        for obj in objs["objs"]:
            bbox.time = objs["timestamp"]
            bbox.uav_id = objs["uav_id"]
            bbox.camera_pose = objs["camera_params"]["pose"]
            bbox.camera_K = objs["camera_params"]["K"]
            bbox.camera_distortion =objs["camera_params"]["distortion"]
            bbox.Bbox = obj["bbox"]
            bbox.class_id = obj["cls_id"]
            bbox.class_name = obj["tracker_id"]
            bbox.uav_pos = obj["loc"]
            packages.append(bbox.copy())
            
        return True
