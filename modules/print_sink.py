from framework import Package
from framework import Sink
from tools import UWConvert

class PrintSink(Sink):
    def __init__(self,offset=None):
        super().__init__("print_sink")
        if offset:
            self.convert = UWConvert(offset)
    def close(self):
        # 对齐操作
        pass

    def process(self, data: Package):
        send_data = {}
        send_data["time"] = data.time
        send_data["obj_cnt"] = 1
        if self.convert:
            data.location[:2] = self.convert.U2W(data.location[:2])
        send_data["objs"] = [{"uav": data.uav_id ,"id": data.global_id, "uid": data.uid, "tr_id": data.tracker_id,  "local_id": data.local_id, "cls": data.class_id,
                              "gis": data.location, "obj_img": data.obj_img}]
        print(send_data)
