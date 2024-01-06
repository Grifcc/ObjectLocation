from framework import *
import time
from multiprocessing import Queue
from . import MQTTSource, TimeFilter, SpatialFilter, EstiPosition


class Process_data(Module):
    def __init__(self, stages_params: list[dict]):
        super().__init__("main_process", None)
        self.process_queue = TimePriorityQueue()

        self.time_slice = stages_params[0]["time_slice"]
        
        self.stage1 = TimeFilter(
            stages_params[0]["time_slice"], stages_params[0]["max_queue_length"])
        self.stage2 = EstiPosition(stages_params[1]["map_path"], stages_params[1]["default_height"],
                                   stages_params[1]["order"], stages_params[1]["enable_reloaction"], stages_params[1]["max_queue_length"])
        self.stage3 = SpatialFilter(stages_params[2]["time_slice"], stages_params[2]["distance_threshold"],
                                    stages_params[2]["max_map"], stages_params[2]["max_queue_length"])

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            if q_in.empty():
                continue
            data = q_in.get()

            for package in data:
                self.process_queue.push(package)

            packages = self.process()

            while q_out.full():
                time.sleep(0.1)
            
            q_out.put(packages)
            print("q_out size:", q_out.qsize())

    def process(self) -> list[Package]:

        if self.process_queue.is_empty() or self.process_queue.delta_time() < self.time_slice + 1:
            return []
        packages = self.process_queue.get_time_slice(self.time_slice)

        if len(packages) == 0:
            return []

        t0 = time.time()
        s1_data = self.stage1.process(packages)
        t1 = time.time()
        print("stage1 time:", t1 - t0)
        
        
        for idx, data in enumerate(s1_data):
            self.stage2.process(s1_data[idx])
        t2 = time.time()
        print("stage2 time:", t2 - t1)
        
        s3_data = self.stage3.process(s1_data)
        t3 = time.time()
        print("stage3 time:", t3 - t2)
        return s3_data
