from tracker import Sort
import json
import numpy as np

def read_log(path):
    """
    Read log file
    """
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    data = [i for i in data if i['obj_cnt'] != 0]
    return sorted(data, key=lambda x: x['time'])


if __name__ == "__main__":
    # Read log file
    data = read_log("data\\20231222_21h35m35s_TH7923461373.txt")
    # Create tracker
    tracker = Sort(20, 3, 5)  # max age, min hits, dis threshold
    # Track points
    tracklets = []
    for idx,val in enumerate(data):
        points = []
        for j in val["objs"]:
            points.append([j["pos"]["latitude"], j["pos"]["longitude"], j["pos"]["altitude"]])
        tracked_points = tracker.update(np.array(points).reshape(-1, 3))
        if tracked_points.size == 0:
            continue
        tracklets.append(tracked_points)
        for i,obj in enumerate(val["objs"]):
            data[idx]["objs"][i]["sort_track_id"] = tracked_points[i][3]
        
    
    # Save tracklet
    print("Done!")
