from tools import MqttClient, Log
import signal


LOG_ROOT = "./log"

broker_url = "192.168.31.158"
port = 1883
client_id = 'record_camera_param'
qos = 2
topic_uav_sn = "thing/product/sn"
timeout = 30


my_log = Log(LOG_ROOT, enable=True, eveytime=False)


def print_writeonce(msg):
    print(msg)

def print_writevertime(msg):
    print(msg["time"], msg["obj_cnt"])


my_log.log_show(print_writeonce)

# 创建MQTT客户端实例,不记录日志
client = MqttClient(broker_url=broker_url,
                    port=port,
                    client_id=client_id,
                    qos=qos,
                    topic_uav_sn=topic_uav_sn,
                    timeout=timeout,
                    log=my_log)

client.start()


def signal_handler(signum, frame):
    print(
        "Ctrl+C detected. Closing socket. Exiting...")
    client.close()

    exit()


signal.signal(signal.SIGINT, signal_handler)

while True:
    client.get_data()
