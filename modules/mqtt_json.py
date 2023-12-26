from framework import Package
from framework import Source
from tools import  ParseMsg
import json
class MQTTJsonSource(Source):
    def __init__(self, offset="data/offset.txt", bbox_type="xywh", json_file="data/log.json"):
        super().__init__("mqtt_json_source")
        self.parse_msg = ParseMsg(offset, bbox_type)
        with open(json_file, "r") as f:
            self.json_data = json.load(f)["content"]
            print("json data length: ", len(self.json_data))

    def close(self):
        # 停止客户端订阅
        pass

    def process(self, packages: list[Package]):
        if len(self.json_data) == 0:
            return False
        objs = self.json_data.pop(0)
        package = self.parse_msg.parse_msg(objs)
        packages.extend(package)
        return True
