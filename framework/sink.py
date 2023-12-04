from .module import *


class Sink(Module):
    def __init__(self, name, max_queue_length=None):
        super().__init__(name, max_queue_length)

    def process(self, package: Package):
        return NotImplementedError

    def run(self):
        while True:
            self.input_lock.acquire()
            if self.input_queue.is_empty():
                self.input_lock.release()
                continue

            package = self.input_queue.pop()
            self.input_lock.release()

            self.process(package)
