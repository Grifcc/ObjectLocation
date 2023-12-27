from framework import Package
from framework import Tracker

from tracker import Sort
import numpy as np


class MOTracker(Tracker):
    def __init__(self, max_age=10, min_hits=3, distance_threshold=4, max_queue_length=None):
        super().__init__("MOTracker", max_queue_length)
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracker = {}

    def new_tracker(self):
        return Sort(max_age=self.max_age, min_hits=self.min_hits, distance_threshold=self.distance_threshold)

    def process(self, data: list[Package]):
        
        point_map = {}
        for package in data:
            if package.uav_id not in self.tracker.keys():
                self.tracker[package.uav_id] = self.new_tracker()
            if package.uav_id not in point_map.keys():
                point_map[package.uav_id] = []
            point_map[package.uav_id].append(package)   
        
        for uav_id,packages in point_map.items():
            points = []
            for p in packages:
                points.append([*p.location, p.class_id])
            points = np.array(points).reshape(-1, 4)
            ret = self.tracker[uav_id].update(points)
            for i, t in enumerate(ret):
                point_map[uav_id][i].tracker_id = t[4]     

        return_data = []
        for _,v in point_map.items():
            return_data.extend(v)
        return return_data
