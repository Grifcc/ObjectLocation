from .__init__ import *
from multiprocessing import Process, Queue


class Pipeline_mp:
    def __init__(self, modules):
        self.modules = modules
        self.queue_list = []
        self.process_pool = []
        self.build()

    def build(self):
        for stages in range(len(self.modules)):
            self.queue_list.append(Queue(maxsize=10000))
            for m in range(len(self.modules[stages])):
                if stages != 0:
                    self.process_pool.append(Process(
                        target=self.modules[stages][m].run_by_process, args=(self.queue_list[stages-1], self.queue_list[stages],)))
                else:
                    self.process_pool.append(Process(
                        target=self.modules[stages][m].run_by_process, args=(self.queue_list[stages],)))

    def run(self):
        for i in self.process_pool[1:]:
            i.start()

        # for p in self.process_pool[1:]:
        #     p.join()
         
        self.process_pool[0].run()   
        print("All processes completed")
