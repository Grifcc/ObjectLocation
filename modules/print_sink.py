from framework import Package
from framework import Sink
import requests
import time


class PrintSink(Sink):
    def __init__(self, url, max_retries=5):
        super().__init__("http_sink")
        self.url = url
        self.max_retries = max_retries

    def process(self, data: Package):
        retry_count = 0
        send_data = {}
        send_data["time"] = data.time
        send_data["obj_cnt"] = 1
        send_data["objs"] = [{"id": data.global_id, "cls": data.cls,
                              "gis": data.location, "obj_img": data.obj_img}]

        print(send_data)
