from framework import Package
from framework import Filter
import numpy as np

# global溯源表循环队列
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.size = 0
        self.front = 0
        self.rear = -1

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def enqueue(self, item):
        if self.is_full():
            # 如果队列满了，删除最旧的元素
            self.dequeue()

        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            return None

        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

    def display(self): # TODO 需优化
        print("Current Queue: ", end="")
        index = self.front
        for _ in range(self.size):
            print(self.queue[index], end=" ")
            index = (index + 1) % self.capacity
        print()

    def get_last_element(self):
        if self.is_empty():
            return None
        return self.queue[self.rear]


class SpatialFilter(Filter):
    def __init__(self, time_slice, max_queue_length=None):
        super().__init__("SpatialFilter", time_slice, max_queue_length)
        self.global_history = self.create_history(20)
        self.distance_threshold = None # 超参
    
    def create_history(self, max_number):
        self.global_history = CircularQueue(max_number)
        self.global_history.enqueue([{}]) # TODO 这个代码history为null时报错，所以先enqueue


    # 设置阈值
    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold = distance_threshold

    # 按照class_id为两个列表，每个列表再按照uav_id分行
    def classify_classid_uav(self, packages: list[Package]):
        class0_list = [package for package in packages if package.class_id == 0]
        class1_list = [package for package in packages if package.class_id == 1]
    
        from collections import defaultdict
        group1 = defaultdict(list)
        for package in class0_list:
            group1[package.uav_id].append(package)
        class0_list = list(group1.values())

        group2 = defaultdict(list)
        for package in class1_list:
            group2[package.uav_id].append(package)
        class1_list = list(group2.values())

        return class0_list, class1_list

    # 空间滤波1:赋值local_id(将不同相机间距离相近的点视为同一空间点，使用相同local_id)
    '''
    注意:相机内的local_id不同，两个相机间同个local_id最多有两个点
    输入:distance_threshold, detections_list (距离阈值，观测数据)
    输出:具有local_id的detections_list
    '''
    def Spatial_filter1(self, distance_threshold, detections_list, local_id=0):
        # 讲各个相机的观测数据进行local_id，如果相机间观测的位置很接近，认为同一目标。
        for i in range(len(detections_list)): # uav_id
            list_i = detections_list[i]
            for j in range(i+1, len(detections_list)): # uav_id+1
                list_j = detections_list[j]

                # 创建两个列表的距离矩阵，并初始化为最大值
                matrix_distance = np.full((len(list_i), len(list_j)), float('inf'))
                # 将距离小于阈值的赋值即可
                for index_i, child_list_i in enumerate(list_i):
                    if child_list_i.local_id is None:
                        child_list_i.local_id = local_id 
                        local_id = local_id+1
                    for index_j, child_list_j in enumerate(list_j): # 找到与child_list_i距离最小的下标（list_j中）
                        distance = np.linalg.norm(np.array(child_list_i.location) - np.array(child_list_j.location))
                        if distance <distance_threshold: # 如果距离小：更新距离和下标
                            matrix_distance[index_i, index_j] = distance

                # 找到矩阵中的最小值及其索引,找到共视点，更新local_id
                for i in range(min(len(list_i),len(list_j))):
                    min_index = np.argmin(matrix_distance)
                    min_index_2d = np.unravel_index(min_index, matrix_distance.shape) #(a,b)说明list_i的第a个与list_j的第b个距离最近
                    list_j[min_index_2d[1]].local_id = list_i[min_index_2d[0]].local_id
                    matrix_distance[i][:] = float('inf')
                    matrix_distance[:][j] = float('inf')
                
        # 更新最后一个uav列表的local_id                   
        for child_list_i in detections_list[-1]:
            if child_list_i.local_id is None:
                child_list_i.local_id = local_id 
                local_id = local_id+1         

        # for i in range(len(detections_list)):
        #     for j in range(len(detections_list[i])):
        #         detect = detections_list[i][j]
        #         print(f"Time: {detect.time},uav_id:{detect.uav_id}, Track ID: {detect.track_id}, local_id:{detect.local_id},  uv:{detect.boundingbox}, point:{detect.pose_location}")
        #     print()

        return detections_list, local_id


    # 空间滤波2:根据local_id更新平均距离，同local_id会按照track_id排序
    '''
    输入:detections_list (观测数据)
    输出:更新距离后的detections_list
    '''
    def Spatial_filter2(self, detections_list1, detections_list2):
        from collections import defaultdict
        # 拉成一维
        detections_list = [item for sublist in detections_list1 for item in sublist] + [item for sublist in detections_list2 for item in sublist]       
        grouped_detections = defaultdict(list) # 这里与普通的字典不同，这里与原数据引用的是同一块，会同时改变
        # 使用 defaultdict 初始化一个字典，键为 local_id，值为包含相同 local_id 的元素的列表
        for detect in detections_list:
            grouped_detections[detect.local_id].append(detect)
        
        # 更新距离: 遍历列表，累加相同 local_id 的坐标和计数,并求解平均距离
        for local_id, group in grouped_detections.items():
            # print(f"Local ID: {local_id}")
            sum_pose_location = np.array([0., 0., 0.])
            for detect in group:
                sum_pose_location += detect.pose_location
            aver_pose_location = sum_pose_location / len(group)
            for detect in group:
                detect.pose_location = aver_pose_location
            group = sorted(group, key=lambda x: x.uav_id)

        #     for detect in group:
        #         print(f"Time: {detect.time}, uav_id:{detect.uav_id},global_id:{detect.global_id}, Track ID: {detect.track_id}, local_id:{detect.local_id},  uv:{detect.boundingbox}, point:{detect.pose_location}")
        # for detect in detections_list:
        #     print(f"Time: {detect.time}, uav_id:{detect.uav_id},global_id:{detect.global_id}, Track ID: {detect.track_id}, local_id:{detect.local_id},  uv:{detect.boundingbox}, point:{detect.pose_location}")
        return grouped_detections

    
    # 追溯global
    '''
    输入:grouped_detections, global_queue(空间过滤后的detection, global溯源表)
    输出:grouped_detections, global_queue(更新global后的detection, 更新后global溯源表)
    '''
    def find_global(grouped_detections, global_queue):
        last_track_list= global_queue.get_last_element() # eg.last_track_list[i]为 global_id_index - {uav1:track2, uav3:track2}
        new_list = [{}]*len(last_track_list) 
        global_id = len(last_track_list) 
        for local_id, group in grouped_detections.items():
            # print(f"Local ID: {local_id}")
            # 找到当下loacl_id之前追踪到的点
            found = False    
            for detect in group:
                detect_track_id = detect.track_id
                detect_uav_id = detect.uav_id
                for i in range(len(last_track_list)):
                    if detect_uav_id in last_track_list[i] and detect_track_id == last_track_list[i][detect_uav_id]:
                        found = True
                        my_dict = {}
                        for detect in group:
                            detect.global_id = i # 更新global_id
                            # 更新global溯源表
                            my_dict[detect.uav_id] = detect.track_id 
                        new_list[i] = my_dict
                        break
                if found:
                    break
            if(not found):
                my_dict = {}
                for detect in group:
                    detect.global_id = global_id # 更新global_id
                    # 更新global溯源表
                    my_dict[detect.uav_id] = detect.track_id 
                new_list.append(my_dict)
                global_id = global_id+1
        
        global_queue.enqueue(new_list) # 更新溯源表
        detections_list = []
        for group in grouped_detections.values():
            detections_list.extend(group)
        return detections_list, global_queue
    
    def process(self, packages: list[Package]):
        # 拆解list，便于后续操作
        class0_list, class1_list = self.classify_classid_uav(packages)
        # 赋值local_id
        class0_list, local_id = self.Spatial_filter1(self.distance_threshold, class0_list, local_id=0)
        class1_list, local_id = self.Spatial_filter1(self.distance_threshold, class1_list, local_id=local_id)
        # 空间滤波2:根据local_id更新平均距离，同local_id会按照track_id排序
        group_list = self.Spatial_filter2(class0_list, class1_list)
        # global_id溯源
        packages, self.global_history = self.find_global(group_list, self.global_history)
 


