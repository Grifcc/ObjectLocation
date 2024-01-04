from tools import MqttClient, Log
import signal
import json
import time

LOG_ROOT = "./log"

broker_url = "127.0.0.1"
port = 1883
client_id = 'record_camera_param'
qos = 2
topic_uav_sn = "thing/product/sn"
timeout = 30


def topic_uav_msg(sn):
    return f"thing/product/{sn}/target_state"


# 创建MQTT客户端实例,不记录日志
client = MqttClient(broker_url=broker_url,
                    port=port,
                    client_id=client_id,
                    qos=qos,
                    topic_uav_sn=topic_uav_sn,
                    timeout=timeout)

client.start()


def signal_handler(signum, frame):
    print(
        "Ctrl+C detected. Closing socket. Exiting...")
    client.close()

    exit()


signal.signal(signal.SIGINT, signal_handler)


log_path = "log_sqr/20231226_16h19m31s_150.log"
with open(log_path, "r") as f:
    json_data = [json.loads(line) for line in f.readlines()]


sn_list = []

for data in json_data:
    if data["uav_id"] not in sn_list:
        sn_list.append(data["uav_id"])
        client.publish(topic_uav_sn, json.dumps({"sn": data["uav_id"]}))
        print(f"MQTT: publish {topic_uav_sn} success!")
        time.sleep(0.1)

time.sleep(5)

for data in json_data:
    client.publish(topic_uav_msg(data["uav_id"]), json.dumps(data))
    time.sleep(0.1)

print("publish finished!")
