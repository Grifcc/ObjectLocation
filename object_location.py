from modules.json_source import JsonSource
from modules.spatial_filter import SpatialFilter
from modules.time_filter import TimeFilter
from modules.print_sink import PrintSink
from modules.esti_position import EstiPosition

from framework.pipeline import Pipeline   

if __name__ == "__main__":
    modules=[]
    modules.append(JsonSource("simulated_data/simulate_data.json"))
    modules.append(TimeFilter(100))
    modules.append(EstiPosition("data/JiulongLake.obj"))
    modules.append(SpatialFilter(100, distance_threshold=2.,max_map=50))
    modules.append(PrintSink("http://192.168.31.31:8888/jk-ivas/non/controller/postTarPos.do"))

    pipe= Pipeline(modules)
    pipe.run()
