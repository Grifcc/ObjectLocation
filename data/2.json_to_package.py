# json转为class package
import json

class Package:
    def __init__(self, time):

        # Public variables
        # read only
        self.time = time
        self.uav_id: int = None
        self.camera_pose: list[float] = [] # [yaw,pitch,roll,x,y,z]
        self.camera_K: list[float] = [] # [fx,fy,cx,cy]
        self.camera_distortion: list[float] = [] #[k1,k2,p1,p2]
        self.Bbox: list[int] = []
        self.class_id: int = None
        self.class_name: str = None
        self.tracker_id: int = None
        self.uav_pos: list[float] = []
        self.obj_img: str = None
        # read & write
        self.global_id: int = None
        self.local_id: int = None
        self.location: list[float] = []

    def get_center_point(self) -> list[float]:
        return [(self.Bbox[0]+self.Bbox[2])/2, (self.Bbox[1]+self.Bbox[3])/2] # TODO 有错误

    def __str__(self):
        return f"time:{self.time}"
    
    def display_info(self):
        print(f"Time: {self.time}")
        print(f"UAV ID: {self.uav_id}")
        print(f"Camera Pose: {self.camera_pose}")
        print(f"Camera K: {self.camera_K}")
        print(f"Camera Distortion: {self.camera_distortion}")
        print(f"Bbox: {self.Bbox}")
        print(f"Class ID: {self.class_id}")
        print(f"Class Name: {self.class_name}")
        print(f"Tracker ID: {self.tracker_id}")
        print(f"UAV Position: {self.uav_pos}")
        print(f"Object Image: {self.obj_img}")
        print(f"Global ID: {self.global_id}")
        print(f"Local ID: {self.local_id}")
        print(f"Location: {self.location}")

def parse_json_to_packages(file_path):
    packages = []

    with open(file_path, 'r') as file:
        data = json.load(file)

        # 解析JSON数据并创建Package对象
        for obj in data['objs']:
            package = Package(data['timestamp'])

            package.uav_id = data['uav_id']
            package.camera_pose = data['camera_params']['pose']
            package.camera_K = data['camera_params']['K']
            package.camera_distortion = data['camera_params']['distortion']
            package.Bbox = obj['bbox']
            package.class_id = obj['cls_id']
            package.tracker_id = obj['tracker_id']
            package.obj_img = data['obj_img']

            # 可以根据需要设置其他属性...

            packages.append(package)

    return packages

# 示例路径
file_path = 'jsons/time1.json'

# 解析JSON文件并创建Package对象列表
packages = parse_json_to_packages(file_path)

# 打印每个Package对象的信息
for package in packages:
    package.display_info()
