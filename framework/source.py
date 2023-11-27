from .module import *
import time


class Source(Module):
    def __init__(self, name, max_queue_length=None):
        super().__init__(name, None,  max_queue_length)

    def process(self, package: Package):
        return NotImplementedError

    def run(self):
        while True:
            package = Package()
            if not self.process(package):
                break
            while self.output_queue.is_full():
                time.sleep(0.1)
            self.output_lock.acquire()
            self.output_queue.push(package)
            self.output_lock.release()
            
