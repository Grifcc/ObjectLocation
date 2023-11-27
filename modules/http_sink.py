from framework import Sink
from framework import Package
import requests
import time


class HttpSink(Sink):
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

        while retry_count < self.max_retries:
            response = requests.post(self.url, data=send_data)
            if response.status_code == 200 and eval(response.text)["resCode"] == 1:
                return
            else:
                retry_count += 1
                time.sleep(0.5)

        raise TimeoutError("Max retries exceeded")

    def __str__(self):
        return "HttpSink: " + self.name + " " + self.url
