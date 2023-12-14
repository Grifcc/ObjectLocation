from modules.json_source import JsonSource
from modules.spatial_filter import SpatialFilter
from modules.time_filter import TimeFilter
from modules.unity_sink import UnitySink
from modules.esti_position import EstiPosition

from framework.pipeline import Pipeline
import signal
import sys

if __name__ == "__main__":

    modules = []
    modules.append(JsonSource("simulated_data/simulate_data.json"))
    modules.append(TimeFilter(1000))
    modules.append(EstiPosition("data/JiulongLake.obj"))
    modules.append(SpatialFilter(1000, distance_threshold=10., max_map=10))
    modules.append(UnitySink(port=8011))
    pipe = Pipeline(modules)
    pipe.run()
    signal.signal(signal.SIGINT, lambda signal, frame: (print(
        "Ctrl+C detected. Closing socket. Exiting..."), modules[-1].close(), exit()))
