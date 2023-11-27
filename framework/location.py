from .module import *
import time


class Location(Module):
    def __init__(self, name, input_queue, output_queue, max_queue_length=None):
        super().__init__(name, input_queue, output_queue, max_queue_length)

    def process(self, package:Package):
        return NotImplementedError

    def run(self):
        while True:
            if self.input_queue.is_empty():
                continue
            try:
                package = self.input_queue.pop()
            except IndexError:
                continue

            self.process(package)
            while self.output_queue.is_full():
                time.sleep(0.1)
            self.output_queue.push(package)
