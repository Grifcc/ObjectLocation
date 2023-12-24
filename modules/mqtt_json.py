from framework import Package
from framework import Source
import numpy as np
from paho.mqtt import client as mqtt_client
from data import DISTORTION_MAP
import math
from tools import UWConvert
import time
import os
import json


class MQTTJsonSource(Source):
    def __init__(self, offset="data/offset.txt", bbox_type="xywh", json_file="data/log.json"):
        super().__init__("mqtt_json_source")

        self.convert = UWConvert(offset)
        self.bbox_type = bbox_type
        with open(json_file, "r") as f:
            self.json_data = json.load(f)["content"]
            print("json data length: ", len(self.json_data))

    def parse_pose(self, c_pose: list):
        pose = []
        utm_point = self.convert.W2U(
            [c_pose[0]["latitude"], c_pose[0]["longitude"],c_pose[0]["altitude"]])
        pose.append(c_pose[1]["yaw"])
        pose.append(c_pose[1]["pitch"])
        pose.append(c_pose[1]["roll"])
        pose.append(utm_point)
        return pose

    def parse_K_D(self, params: dict):
        distortion = DISTORTION_MAP[params["focal_type"]]
        fov_h = params['fov'][0]
        fov_v = params['fov'][1]
        resolution_x = params['resolution'][0]
        resolution_y = params['resolution'][1]
        cx = resolution_x / 2
        cy = resolution_y / 2
        fx = cx * math.tan(math.radians(fov_h)/2.)
        fy = cy * math.tan(math.radians(fov_v)/2.)
        return [fx, fy, cx, cy], distortion

    def parse_bbox(self, bbox, resolution):
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["w"]
        h = bbox["h"]
        return [x*resolution[0], y*resolution[1], w*resolution[0], h*resolution[1]]

    def close(self):
        # 停止客户端订阅
        pass

    def process(self, packages: list[Package]):

        if len(self.json_data) == 0:
            return False
        objs = self.json_data.pop(0)
        if objs["obj_cnt"] == 0:
            return True
        for idx, obj in enumerate(objs["objs"]):
            if obj["cls_id"] not in [2, 3, 4, 5, 6]:
                continue
            bbox = Package()
            bbox.bbox_type = self.bbox_type
            bbox.time = objs["time"]
            bbox.uav_id = objs["uav_id"]
            bbox.camera_id = objs["camera_id"]
            bbox.camera_pose = self.parse_pose(
                [objs['global_pos'], objs['camera']])
            bbox.camera_K, bbox.camera_distortion = self.parse_K_D(
                objs['camera'])
            bbox.norm_Bbox = [obj["bbox"]["x"], obj["bbox"]["y"],
                                  obj["bbox"]["w"], obj["bbox"]["h"]]
            bbox.Bbox = self.parse_bbox(
                obj["bbox"], objs['camera']['resolution'])
            bbox.class_id = obj["cls_id"]
            bbox.tracker_id = obj["track_id"]
            bbox.uav_wgs = [obj["pos"]["latitude"],
                            obj["pos"]["longitude"], obj["pos"]["altitude"]]
            bbox.uav_utm = self.convert.W2U(bbox.uav_wgs)
            bbox.obj_img = None if obj['pic'] == "None" else obj['pic']

            packages.append(bbox.copy())
        return True
