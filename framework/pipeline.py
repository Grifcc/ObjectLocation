from .__init__ import *
import threading


class Pipeline:
    def __init__(self, modules: list[Module]):
        self.modules = modules
        self.queue_list = []
        self.thread_pool = []
        self.lock_list: list[threading.Lock] = []
        self.build()

    def build(self):
        for i in range(len(self.modules)):
            self.queue_list.append(TimePriorityQueue())
            self.lock_list.append(threading.Lock())

            self.modules[i].set_output_queue(self.queue_list[i])
            self.modules[i].set_output_lock(self.lock_list[i])
            if i != 0:
                self.modules[i].set_input_queue(self.queue_list[i-1])
                self.modules[i].set_input_lock(self.lock_list[i-1])

        for i in range(len(self.modules)):
            self.thread_pool.append(
                threading.Thread(target=self.modules[i].run))

    def run(self):
        for i in self.thread_pool:
            i.start()
