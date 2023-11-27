from .module import *

class Sink(Module):
    def __init__(self, name, input_queque, max_queue_length=None):
        super().__init__(name, input_queque, None, max_queue_length)

    def process(self, package: Package):
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
