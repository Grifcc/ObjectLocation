from framework import Package
from framework import Sink
import requests
import time
import json
from tools import UWConvert
from multiprocessing import Queue


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

    def process(self, data: list[Package]):
        retry_count = 0
        send_data = {}
        send_data["timestamp"] = data[0].time
        send_data["obj_cnt"] = len(data)
        send_data["objs"] = []

        for obj in data:
            if self.convert:

                obj.location = self.convert.U2W(obj.location)
            send_data["objs"].append({"id": obj.global_id, "cls": obj.class_id,
                                      "gis": obj.location, "bbox": obj.norm_Bbox,
                                      "obj_img": f"http://192.168.31.210:9002/detect/{obj.obj_img}.jpg" if obj.obj_img else "null"})

        send_data = json.dumps(send_data)

        while retry_count < self.max_retries:
            response = requests.post(
                self.url, data=send_data, headers=self.header)
            if response.status_code == 200 and eval(response.text)["resCode"] == 1:
                print(f"\033[92mHttp:{data[0].time}  {len(data)}\033[0m")
                return
            else:
                retry_count += 1
                time.sleep(0.5)

        raise TimeoutError("Max retries exceeded")

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            # print("q_in size:", q_in.qsize())
            if q_in.empty():
                continue
            data = q_in.get()
            if len(data) == 0:
                continue
            same_time_data = []
            for package in data:
                if same_time_data == []:
                    same_time_data = [package]
                elif package.time != same_time_data[0].time:
                    self.process(same_time_data)
                    same_time_data = []

                same_time_data.append(package)
