from framework import Package
from framework import Sink
import socket
import time
import copy
import json


class UnitySink(Sink):
    def __init__(self, hostip='127.0.0.1', port=8888, max_retries=5, buffsize=1024,):
        super().__init__("unity_sink")
        self.max_retries = max_retries
        self.address = (hostip, port)
        self.buffsize = buffsize
        self.encoding = 'utf-8'
        self.server_socket = None
        self.client_socket = None

        try:
            # 创建 TCP 服务器
            self.server_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(self.address)
            self.server_socket.listen(1)
            # 与客户端建立连接
            while True:
                print('Waiting for connect...')
                # 等待客户端连接
                self.client_socket, address = self.server_socket.accept()
                print(
                    f"[*] Accepted connection from {address[0]}:{address[1]}")
                time.sleep(0.1)
                break
        except Exception as e:
            print(e)
            exit(0)

        self.data_tem = {
            'ids': 0,
            'x': 0,
            'y': 0,
            'z': 0,
        }

    def is_socket_connected(self):
        try:
            error_code = self.client_socket.getsockopt(
                socket.SOL_SOCKET, socket.SO_ERROR)
            if error_code == 0:
                return True
        except socket.error:
            pass
        return False

    def close(self):
        try:
            self.client_socket.close()
            self.server_socket.close()
        except:
            pass

    def send_data(self, data: str):
        try:
            self.client_socket.sendall(data.encode())
            return True
        except:
            return False

    def process(self, data: Package):
        # assert self.is_socket_connected(), "Socket 连接错误"
        retry_count = 0
        send_data = copy.deepcopy(self.data_tem)
        send_data["ids"] = data.global_id
        send_data["x"] = float(data.location[0])
        send_data["y"] = float(data.location[1])
        send_data["z"] = float(data.location[2])
        send_str = json.dumps(send_data)
        while retry_count < self.max_retries:
            if not self.send_data(send_str):
                retry_count += 1
                time.sleep(0.01)
            else:
                retry_count = 0
                return
        self.close()
        raise TimeoutError("Max retries exceeded")
