from typing import Union
import threading
import copy

# package包


class Package:
    def __init__(self, time=None):

        # Public variables
        # read only
        self.time = time
        self.uav_id: int = None
        self.camera_pose: list[float] = []  # [yaw,pitch,roll,x,y,z]
        self.camera_K: list[float] = []  # [fx,fy,cx,cy]
        self.camera_distortion: list[float] = []  # [k1,k2,p1,p2]
        self.Bbox: list[int] = []  # [x,y,w,h]
        self.class_id: int = None
        self.class_name: str = None
        self.tracker_id: int = None
        self.uav_pos: list[float] = []
        self.obj_img: str = None
        # read & write
        self.global_id: int = None
        self.local_id: int = None
        self.location: list[float] = []  # (WGS84）

    def get_center_point(self) -> list[float]:
        # TODO 有错误 1车0人
        if self.class_id == 1:
            return [self.Bbox[0]+self.Bbox[2]/2., self.Bbox[1]+self.Bbox[3]/2.]
        elif self.class_id == 0:
            return [self.Bbox[0]+self.Bbox[2]/2., self.Bbox[1]+self.Bbox[3]]

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"time:{self.time}"

# 基于时间优先级的队列   队尾是老时间，队头是新时间
class TimePriorityQueue:
    def __init__(self, max_count=None):
        self._queue: list[Package] = []
        self.__index = 0
        self.max_count = max_count

    # 判断是否为空队列
    def is_empty(self):
        return len(self._queue) == 0
    # 判断队列是否满

    def is_full(self):
        return self.max_count and self.__len__() >= self.max_count

    def push(self, package: Package):
        if self.max_count and self.__len__() >= self.max_count:
            return -1
        idx = 0
        while idx < self.__len__():
            if self._queue[idx].time < package.time:
                self._queue.insert(idx, package)
                break
            idx += 1
        else:
            self._queue.append(package)

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

    # 最大容量
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

    def __getitem__(self, index):
        return self._queue[index]

    def __setitem__(self, index, value):
        self._queue[index] = value


class Module:
    def __init__(self, name, max_queue_length=None):
        self.name: str = name
        self.input_queue: TimePriorityQueue = None
        self.output_queue: TimePriorityQueue = None

        self.input_lock = None
        self.output_lock = None

        self.max_queue_length = max_queue_length
        self._is_running = False

    def set_input_lock(self, lock: threading.Lock):
        self.input_lock = lock

    def set_output_lock(self, lock:  threading.Lock):
        self.output_lock = lock

    def set_input_queue(self, input_queue: TimePriorityQueue):
        self.input_queue = input_queue

    def set_output_queue(self, output_queue: TimePriorityQueue):
        self.output_queue = output_queue
        if self.max_queue_length != None and output_queue != None:
            self.output_queue.set_max_count(self.max_queue_length)

    def run(self):
        return NotImplementedError

    def __str__(self):
        return self.name
