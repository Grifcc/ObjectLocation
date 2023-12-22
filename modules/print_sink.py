from framework import Package
from framework import Sink
from tools import UWConvert

import time
from datetime import datetime
import copy


class PrintSink(Sink):
    def __init__(self, time_freq=5, offset=None):
        super().__init__("print_sink")
        if offset:
            self.convert = UWConvert(offset)
        self.buffer = []
        self.last_time = int(time.time() * 1000)
        self.time_interval = int(1000.0 / time_freq)  # ms

    def close(self):
        # 对齐操作
        pass

    def process(self, data: Package):

        send_data = {}
        send_data["time"] = data.time
        send_data["obj_cnt"] = 1
        if self.convert:
            data.location[:2] = self.convert.U2W(data.location[:2])
        send_data["objs"] = [{"id": data.global_id, "tr_id": data.tracker_id,  "cls": data.class_id,
                              "gis": data.location, "uav_pos": data.uav_pos, "obj_img": f"{data.obj_img}.jpg" if data.obj_img else "null"}]
        self.buffer.append(send_data)

        now_time = int(time.time() * 1000)
        if now_time - self.last_time >= self.time_interval:
            print(
                f'\033[92m{datetime.fromtimestamp(now_time/1000.).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\033[0m->', end="")
            print(self.buffer[0])
            if len(self.buffer) > 1:
                print(len(self.buffer))
                self.buffer.pop()
            self.last_time = now_time

    def run(self):
        while True:
            self.input_lock.acquire()
            if self.input_queue.is_empty():
                self.input_lock.release()
                while len(self.buffer) != 0:
                    now_time = int(time.time() * 1000)
                    if now_time - self.last_time >= self.time_interval:
                        print(
                            f'\033[92m{datetime.fromtimestamp(now_time/1000.).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\033[0m->', end="")
                        print(self.buffer.pop())
                        self.last_time = now_time
                continue

            package = self.input_queue.pop()
            self.input_lock.release()

            self.process(package)
