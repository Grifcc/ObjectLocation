from .module import *
import time


class PreProcess(Module):
    def __init__(self, name, input_queue, output_queue, time_slice, max_queue_length=None):
        super().__init__(name, input_queue, output_queue, max_queue_length)
        self.max_queue_length = max_queue_length
        self.time_slice = time_slice
        if max_queue_length != None:
            self.output_queue.set_max_count(max_queue_length)

    def process(self, packages: list[Package]):
        return NotImplementedError

    def run(self):
        while True:
            if self.input_queue.is_empty():
                continue
            packages = self.input_queue.get_time_slice(self.time_slice)
            if len(packages) == 0:
                continue
            self.process(packages)

            for package in  packages:
                while self.output_queue.is_full():
                    time.sleep(0.1)
                self.output_queue.push(package)
