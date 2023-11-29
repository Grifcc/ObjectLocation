from framework import Package
from framework import Filter

class SpatialFilter(Filter):
    def __init__(self, name, time_slice, max_queue_length=None):
        super().__init__(name, time_slice, max_queue_length)
    
    def process(self, packages: list[Package]):
        pass
