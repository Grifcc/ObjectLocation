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

LOG_ROOT = "./log"


class MQTTSource(Source):
    def __init__(self, offset="data/offset.txt", broker_url="192.168.31.158", port=1883, client_id='sub_camera_param', topic_uav_sn="thing/product/sn", timeout=30):
        super().__init__("mqtt_source")

        self.broker_url = broker_url
        self.client_id = client_id
        self.port = port
        self.topic_uav_sn = topic_uav_sn

        self.timeout = timeout   # second

        self.topic_map = {}  # key: sn  value: topic

        self.buffer: dict = []

        self.convert = UWConvert(offset)

        T_plane_gimbal = np.array([[1., 0,  0,  0.11],
                                  [0., 1., 0., 0.0],
                                  [0., 0., 1., 0.05],
                                  [0, 0, 0, 1]]).reshape(4, 4)
        T_gimbal_camera = np.array([[0, 0,  1.0,  0.0],
                                    [1., 0., 0., 0.0],
                                    [0., 1., 0., 0],
                                    [0, 0, 0, 1]]).reshape(4, 4)
        self.T_camera_plane = np.linalg.inv(T_plane_gimbal @ T_gimbal_camera)

        self.log_files = {}   # key: topic  values: file

        # 创建MQTT客户端实例
        self.client = mqtt_client.Client(self.client_id)
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_url, self.port)
        # 订阅话题
        self.client.subscribe(topic_uav_sn)
        # 设置回调函数，用于处理消息接收事件
        self.client.on_message = self.on_message
        # 开始循环订阅
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        age = 0
        while rc != 0:
            print("%dth connect error, return code %d\n", age, rc)
            age += 1
            if age > self.timeout:
                raise TimeoutError("Max retries exceeded")

        print("Connected to MQTT Broker!\n")

    def on_message(self, client, userdata, msg):
        if msg.topic == self.topic_uav_sn:
            json_data = eval(msg.payload.decode())
            sn_from_mqtt = json_data['sn']

            if sn_from_mqtt not in self.topic_map:
                print("sn_from_mqtt: ", sn_from_mqtt)
                topic = f"thing/product/{sn_from_mqtt}/target_state"
                self.topic_map[sn_from_mqtt] = topic
                self.log_files[topic] = os.path.join(
                    LOG_ROOT, f"{time.time()}_{sn_from_mqtt}_log.txt")
                self.client.subscribe(topic)
                print("Subscribe to: ", topic)

        elif msg.topic in self.topic_map.values():
            data = eval(msg.payload.decode())
            with open(self.log_files[msg.topic], "a+", encoding="utf-8") as f:
                f.write(json.loads(data))
            # print(data)
            if data["obj_cnt"] != 0:
                print(data)
                self.buffer.append(data)

    def parse_pose(self, c_pose: list):
        pose = []
        utm_point = self.convert.W2U(
            [c_pose[0]["latitude"], c_pose[0]["longitude"]])
        pose.append(c_pose[1]["yaw"])
        pose.append(c_pose[1]["pitch"])
        pose.append(c_pose[1]["roll"])
        pose.append([*utm_point, c_pose[0]["altitude"]])
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
        self.client.loop_stop()
        for idx in range(len(self.log_files)):
            self.log_files[idx].close()

    def process(self, packages: list[Package]):

        if len(self.buffer):
            objs = self.buffer.pop()
            for obj in objs["objs"]:
                if obj["cls_id"] not in [2, 3, 4, 5, 6]:
                    continue
                bbox = Package()
                bbox.time = objs["time"]
                bbox.uav_id = objs["uav_id"]
                bbox.camera_id = objs["camera_id"]
                bbox.camera_pose = self.parse_pose(
                    [objs['global_pos'], objs['camera']])
                bbox.camera_K, bbox.camera_distortion = self.parse_K_D(
                    objs['camera'])
                bbox.Bbox = self.parse_bbox(
                    obj["bbox"], objs['camera']['resolution'])
                bbox.class_id = obj["cls_id"]
                bbox.tracker_id = obj["track_id"]
                bbox.uav_pos = obj["pos"]
                bbox.obj_img = None if obj['pic'] == "None" else obj['pic']

                packages.append(bbox.copy())

        return True
