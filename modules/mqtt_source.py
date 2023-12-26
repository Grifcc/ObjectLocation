from framework import Package
from framework import Source
import numpy as np
from tools import Log, MqttClient, ParseMsg
import json


LOG_ROOT = "./log/mqtt_source"


class MQTTSource(Source):
    def __init__(self, offset="data/offset.txt",
                 bbox_type="xywh",
                 broker_url="192.168.31.158",
                 port=1883,
                 client_id='sub_camera_param',
                 qos=2,
                 topic_uav_sn="thing/product/sn",
                 timeout=30):
        super().__init__("mqtt_source")
        my_log = Log(LOG_ROOT, enable=True, eveytime=False)
        
        def print_writeonce(msg):
            data = json.loads(msg)
            # 打印黄色
            print("\033[33mMQTT: ", data["time"], data["obj_cnt"], "\033[0m")
            for obj in data["objs"]:
                if obj["pic"] != None:
                    print("\033[32mMQTT: ", obj["pic"], ".jpg\033[0m")
        my_log.log_show(print_writeonce)   
            
        # 创建MQTT客户端实例,不记录日志
        self.client = MqttClient(broker_url=broker_url,
                                 port=port,
                                 client_id=client_id,
                                 qos=qos,
                                 topic_uav_sn=topic_uav_sn,
                                 timeout=timeout,
                                 log=my_log)

        T_plane_gimbal = np.array([[1., 0,  0,  0.11],
                                  [0., 1., 0., 0.0],
                                  [0., 0., 1., 0.05],
                                  [0, 0, 0, 1]]).reshape(4, 4)
        T_gimbal_camera = np.array([[0, 0,  1.0,  0.0],
                                    [1., 0., 0., 0.0],
                                    [0., 1., 0., 0],
                                    [0, 0, 0, 1]]).reshape(4, 4)
        self.T_camera_plane = np.linalg.inv(T_plane_gimbal @ T_gimbal_camera)

        self.parse_msg = ParseMsg(offset, bbox_type)

    def close(self):
        # 停止客户端订阅
        self.client.close()

    def process(self, packages: list[Package]):
        objs = self.client.get_data()
        package = self.parse_msg.parse_msg(objs)
        packages.extend(package)
        return True
