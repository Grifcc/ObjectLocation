import yaml
import sys
import os

from framework.pipeline_mp import Pipeline_mp
import signal
from multiprocessing import Queue


from modules import MQTTSource
from modules import MOTracker
from modules import Process_data
from modules import HttpSink


if __name__ == "__main__":
    # 读取YAML配置文件
    defualt_cfg = "config_for_mp.yaml"
    try:
        if os.path.exists(sys.argv[1]):
            defualt_cfg = sys.argv[1]
        else:
            raise FileNotFoundError
    except:
        print(f"---Using default config {defualt_cfg}---")

    with open(defualt_cfg, 'r', encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    pipelines = []
    for stage in config["pipeline"].values():
        module = eval(stage["name"])
        pipelines.append([
            module(*stage["args"].values()) if stage["args"] else module() for _ in range(config["global"]["stage_num"][stage["idx"]-1])])

    pipe = Pipeline_mp(pipelines)
    
    # def signal_handler(signum, frame):
    #     print(
    #         "Ctrl+C detected. Closing socket. Exiting...")
    #     pipelines[0].close()
    #     pipelines[-1].close()
    #     for i in pipe.process_pool:
    #         i.terminate()
    #     exit()

    # signal.signal(signal.SIGINT, signal_handler)
    
    pipe.run()


