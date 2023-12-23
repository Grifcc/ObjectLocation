from framework import Package
from framework import Sink
import requests
import time
import json
from tools import UWConvert


class HttpSink(Sink):
    def __init__(self, url, offset=None, max_retries=5):
        super().__init__("http_sink")
        self.url = url
        self.max_retries = max_retries
        self.header = {"content-type": "application/json"}
        self.convert = None
        if offset:
            self.convert = UWConvert(offset)

    def close(self):
        # 对齐操作
        pass

    def process(self, data: Package):
        retry_count = 0
        send_data = {}
        send_data["timestamp"] = data.time
        send_data["obj_cnt"] = 1
        if self.convert:
            data.location = self.convert.U2W(data.location)
        send_data["objs"] = [{"id": data.global_id, "cls": data.class_id,
                              "gis": data.location, "obj_img":f"http://192.168.31.210:9002/detect/{data.obj_img}.jpg"  if data.obj_img else "null"}]
        if send_data["objs"][0]["obj_img"] != "null":
            print(send_data["objs"][0]["obj_img"])

        send_data = json.dumps(send_data)

        while retry_count < self.max_retries:
            response = requests.post(
                self.url, data=send_data, headers=self.header)
            if response.status_code == 200 and eval(response.text)["resCode"] == 1:
                return
            else:
                retry_count += 1
                time.sleep(0.5)

        raise TimeoutError("Max retries exceeded")
