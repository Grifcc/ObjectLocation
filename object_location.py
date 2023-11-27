import framework
from modules.http_sink import HttpSink


if __name__ == "__main__":
    modules=[]
    modules.append(HttpSink("http://192.168.31.31:8888/jk-ivas/non/controller/postTarPos.do"))

