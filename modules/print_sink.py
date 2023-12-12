from framework import Package
from framework import Sink
import requests
import time


class PrintSink(Sink):
    def __init__(self):
        super().__init__("print_sink")

    def process(self, data: Package):
        send_data = {}
        send_data["time"] = data.time
        send_data["obj_cnt"] = 1
        send_data["objs"] = [{"id": data.global_id, "uid": data.uid, "tr_id": data.tracker_id,  "local_id": data.local_id, "cls": data.class_id,
                              "gis": data.location, "obj_img": data.obj_img}]
        print(send_data)
