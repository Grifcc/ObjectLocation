from framework import Package
from framework import Source
import socket
import sys
import numpy as np
from paho.mqtt import client as mqtt_client

DISTORTION_MAP = {}


class UDPSource(Source):
    def __init__(self, broker_url="192.168.31.158", client_id='sub_camera_param', port=1883, topic_uav_sn="thing/product/sn"):
        super().__init__("udp_source")

        self.broker_url = broker_url
        self.client_id = client_id
        self.port = port
        self.topic_uav_sn = topic_uav_sn

        self.max_age = 30   # second

        self.topic_map = {}  # key: sn  value: topic

        self.buffer: dict = []

        T_body_gimbal = np.array([[1., 0,  0,  0.11],
                                  [0., 1., 0., 0.0],
                                  [0., 0., 1., 0.05],
                                  [0, 0, 0, 1]]).reshape(4, 4)
        T_gimbal_camera = np.array([[0, 0,  1.0,  0.0],
                                    [1., 0., 0., 0.0],
                                    [0., 1., 0., 0],
                                    [0, 0, 0, 1]]).reshape(4, 4)
        self.T_body_camera = T_body_gimbal @ T_gimbal_camera

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
            if age > self.max_age:
                raise TimeoutError("Max retries exceeded")

        print("Connected to MQTT Broker!\n")

    def on_message(self, client, userdata, msg):
        if msg.topic == self.topic_uav_sn:
            json_data = eval(msg.payload.decode())
            sn_from_mqtt = json_data['sn']
            print("sn_from_mqtt: ", sn_from_mqtt)

            if sn_from_mqtt not in self.topic_map:
                topic = f"thing/product/{sn_from_mqtt}/target_state"
                self.topic_map[sn_from_mqtt] = topic
                self.client.subscribe(topic)
                print("Subscribe to: ", topic)

        elif msg.topic in self.topic_map.values():
            self.buffera.append(eval(msg.payload.decode()))

    def parse_pose(self, c_pose: list):
        pose = []
        uav_pose = np.array(
            [c_pose[0]["longitude"], c_pose[0]["latitude"], c_pose[0]["altitude"]])
        pose.append(c_pose[1]["yaw"])
        pose.append(c_pose[1]["pitch"])
        pose.append(c_pose[1]["roll"])
        pose.append(uav_pose.tolist())
        return pose

    def parse_K_D(self, params: dict):
        distortion = DISTORTION_MAP[params["focal_type"]]
        fov_h = params['fov'][0]
        fov_v = params['fov'][1]
        resolution_x = params['resolution'][0]
        resolution_y = params['resolution'][1]
        fx = 0
        fy = 0
        cx = 0
        cy = 0
        return [fx, fy, cx, cy], distortion

    def close(self):
        # 停止客户端订阅
        self.client.loop_stop()

    def process(self, packages: list[Package]):

        if len(self.buffer):
            objs = self.buffer.pop()
            for obj in objs["objs"]:
                bbox = Package()
                bbox.time = objs["timestamp"]
                bbox.uav_id = objs["uav_id"]
                bbox.camera_id = objs["camera_id"]
                bbox.camera_pose = self.parse_c_pose(
                    [objs['global_pos'], objs['camera']])
                bbox.camera_K, bbox.camera_distortion = self.parse_K_D(
                    objs['camera'])
                bbox.Bbox = obj["bbox"]
                bbox.class_id = obj["cls_id"]
                bbox.tracker_id = obj["track_id"]
                bbox.uav_pos = obj["pos"]
                bbox.obj_img = obj['obj_pic']

                packages.append(bbox.copy())

        return True
