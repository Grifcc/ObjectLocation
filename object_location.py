from modules.json_source import JsonSource
from modules.spatial_filter import SpatialFilter
from modules.time_filter import TimeFilter
from modules.print_sink import PrintSink
from modules.esti_position import EstiPosition

from framework.pipeline import Pipeline   

if __name__ == "__main__":
    modules=[]
    modules.append(JsonSource("data/simulate_data.json"))
    modules.append(TimeFilter(100))
    modules.append(EstiPosition("data/odm_textured_model_geo.obj"))
    modules.append(SpatialFilter("spatial_filter", 0.1))
    modules.append(PrintSink("http://192.168.31.31:8888/jk-ivas/non/controller/postTarPos.do"))


    pipe= Pipeline(modules)
    pipe.run()
