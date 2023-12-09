import copy
import math

objects_attribute = {
    "cls_id": -1,
    "start_point": [],
    "end_point": [],
    "distance": 0,
    "speed": 0,
    "angle": 0
}


def get_angle(p1, p2):
    if p2[0]-p1[0] != 0:
        slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    else:
        if p2[1]-p1[1] > 0:
            return 90
        else:
            return -90
    return math.degrees(math.atan(slope))


def get_distance(p1, p2):
    return math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)


def get_attr(start_point: list[int, int], end_point: list[int, int], duration: float) -> dict:
    """
    根据起点、终点和持续时间计算并返回物体的属性。

    Args:
        start_point (list[int, int]): 物体的起点坐标。
        end_point (list[int, int]): 物体的终点坐标。
        duration (float): 物体从起点到终点的运动持续帧数。

    Returns:
        dict: 包含物体属性的字典：
            - "start_point" (list[int, int]): 起点坐标。
            - "end_point" (list[int, int]): 终点坐标。
            - "angle" (float): 起点和终点之间的移动角度。
            - "distance" (float): 起点和终点之间的距离。
            - "speed" (float): 物体的速度，计算公式为距离除以持续帧数。
            - "cls_id" (int): 物体的类别ID,0代表慢行(人),1代表快速(车辆)。
    """
    attr = copy.deepcopy(objects_attribute)
    attr["start_point"] = start_point
    attr["end_point"] = end_point
    attr["angle"] = get_angle(start_point, end_point)
    attr["distance"] = get_distance(start_point, end_point)
    attr["speed"] = attr["distance"] / duration
    if attr["speed"] > 0.3:
        attr["cls_id"] = 1  # fast 车
        attr["bbox"] = [40, 20]
    else:
        attr["cls_id"] = 0  # slow 人
        attr["bbox"] = [10, 20]

    return attr
