from framework import Package
from framework import Source
import numpy as np
from tools import UWConvert, MqttClient, ParseMsg


LOG_ROOT = "./log"


class MQTTSource(Source):
    def __init__(self, offset="data/offset.txt", bbox_type="xywh", broker_url="192.168.31.158", port=1883, client_id='sub_camera_param', qos=2, topic_uav_sn="thing/product/sn", timeout=30):
        super().__init__("mqtt_source")

        # 创建MQTT客户端实例,不记录日志
        self.client = MqttClient(broker_url=broker_url, port=port, client_id=client_id,
                                 qos=qos, topic_uav_sn=topic_uav_sn, timeout=timeout)

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
