from typing import Union


class Package:
    def __init__(self, time):

        # Public variables
        # read only
        self.time = time
        self.uav_id: int = None
        self.camera_pose: list[float] = []
        self.Bbox: list[int] = []
        self.class_id: int = None
        self.class_name: str = None
        self.tracker_id: int = None
        self.uav_pos: list[float] = []
        # read & write
        self.global_id: int = None
        self.local_id: int = None
        self.location: list[float] = []

    def get_center_point(self) -> list[float]:
        return [(self.Bbox[0]+self.Bbox[2])/2, (self.Bbox[1]+self.Bbox[3])/2]

    def __str__(self):
        return f"time:{self.time}"


class TimePriorityQueue:
    def __init__(self, max_count=None):
        self._queue: list[Package] = []
        self.__index = 0
        self.max_count = max_count

    def is_empty(self):
        return len(self._queue) == 0

    def is_full(self):
        return self.__len__() >= self.max_count and self.max_count != None

    def push(self, package: Package):
        if self.is_empty():
            self._queue.append(package)
            return 1
        if self.__len__() >= self.max_count and self.max_count != None:
            return -1
        for idx, val in enumerate(self._queue):
            if val.time > package.time:
                self._queue.insert(idx, package)
                return 1

    def pop(self):
        if self.is_empty():
            raise IndexError("TimePriorityQueue is empty")
        return self._queue.pop()

    def clear(self):
        self._queue.clear()

    def get_time_slice(self, time_slice):
        stop_idx = None
        if self.is_empty():
            raise IndexError("TimePriorityQueue is empty")
        for idx in range(self.__len__()):
            if self._queue[idx].time - self._queue[0].time > time_slice:
                stop_idx = idx-1
                break
            if idx == self.__len__()-1:
                stop_idx = idx

        if stop_idx == None:
            return []
        time_slice_list = self._queue[:stop_idx]
        self._queue = self._queue[stop_idx:]
        return time_slice_list

    def set_max_count(self, max_count):
        self.max_count = max_count

    def __len__(self):
        return len(self._queue)

    def __str__(self) -> str:
        return str(self._queue)

    def __len__(self):
        return len(self._queue)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self._queue):
            raise StopIteration
        item = self._queue[self.__index]
        self.__index += 1
        return item


class Module:
    def __init__(self, name, input_queue, output_queue, max_queue_length=None):
        self.name: str = name
        self.input_queue: TimePriorityQueue = input_queue
        self.output_queue: TimePriorityQueue = output_queue
        self.max_queue_length =max_queue_length
        if  self.max_queue_length != None and output_queue != None:
            self.output_queue.set_max_count(max_queue_length)
        self._is_running = False

    def run(self):
        return NotImplementedError


if __name__ == "__main__":
    packages = []
    import random
    for i in range(100):
        package = Package(i)
        packages.append(package)

    random.shuffle(packages)
    queue = TimePriorityQueue(50)
    for package in packages:
        print(package.time)
        if queue.push(package) > 0:
            print("push success")
        else:
            print("push fail")

    x = queue.get_time_slice(10)
    for i in x:
        print(i)
    print("===")
    for i in queue:
        print(i)
    print("==123=")
    for i in queue:
        print(i)
