from data import DISTORTION_MAP, CLS_MAP
import math
from framework import Package
from .parse_utm import UWConvert


def parse_K_D(params: dict):
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


def parse_bbox( bbox, resolution):
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]
    return [x*resolution[0], y*resolution[1], w*resolution[0], h*resolution[1]]


class ParseMsg:
    def __init__(self, offset="data/offset.txt", bbox_type="xywh", cls_list: list = [2, 3, 4, 5, 6]):
        self.convert = UWConvert(offset)
        self.bbox_type = bbox_type
        self.cls_list = cls_list

    def parse_pose(self, c_pose: list):
        pose = []
        utm_point = self.convert.W2U(
            [c_pose[0]["latitude"], c_pose[0]["longitude"], c_pose[0]["altitude"]])
        pose.append(c_pose[1]["yaw"])
        pose.append(c_pose[1]["pitch"])
        pose.append(c_pose[1]["roll"])
        pose.append(utm_point)
        return pose

    def parse_msg(self, msg):
        packages = []
        for obj in msg["objs"]:
            if obj["cls_id"] in CLS_MAP.keys():
                cls_id = CLS_MAP[obj["cls_id"]]
            else:
                cls_id = 98  # unknow
            if cls_id not in self.cls_list:
                continue
            bbox = Package()
            bbox.time = msg["time"]
            bbox.uav_id = msg["uav_id"]
            bbox.camera_id = msg["camera_id"]
            bbox.camera_pose = self.parse_pose(
                [msg['global_pos'], msg['camera']])
            bbox.camera_K, bbox.camera_distortion = parse_K_D(
                msg['camera'])
            bbox.Bbox = parse_bbox(
                obj["bbox"], msg['camera']['resolution'])
            bbox.norm_Bbox = [obj["bbox"]["x"], obj["bbox"]["y"],
                              obj["bbox"]["w"], obj["bbox"]["h"]]
            bbox.class_id = cls_id
            bbox.tracker_id = obj["track_id"]
            bbox.uav_wgs = [obj["pos"]["latitude"],
                            obj["pos"]["longitude"], obj["pos"]["altitude"]]
            bbox.uav_utm = self.convert.W2U(bbox.uav_wgs)
            bbox.obj_img = None if obj['pic'] == "None" else obj['pic']

            packages.append(bbox.copy())
        return packages
