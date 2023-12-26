import numpy as np
import scipy
from scipy.spatial.distance import cdist
from .KalmanFilter import KalmanPointTracker
import matplotlib.pyplot as plt


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def distance_mat(points1, points2):
    """
    Return distance matrix between points1 and points2
    """

    if not isinstance(points1, np.ndarray):
        points1 = np.array(points1).reshape(-1, 3)
    if not isinstance(points2, np.ndarray):
        points2 = np.array(points2).reshape(-1, 3)

    return cdist(points1, points2)


def associate_points_to_trackers(points, trackers, distance_threshold=0.3):
    """
    Assigns points to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_points and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(points)), np.empty((0, 4), dtype=int)

    distance_matrix = distance_mat(points, trackers)

    if min(distance_matrix.shape) > 0:
        a = (distance_matrix < distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_points = []
    for d, det in enumerate(points):
        if (d not in matched_indices[:, 0]):
            unmatched_points.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with  large distance
    matches = []
    for m in matched_indices:
        if (distance_matrix[m[0], m[1]] > distance_threshold):
            unmatched_points.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_points), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, distance_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, points=np.empty((0, 3))):
        """
        Params:
        points - a numpy array of points in the format [[x1,y1,z1],[x2,y2,z2],...]
        Requires: this method must be called once for each frame even with empty points (use np.empty((0, 5)) for frames without points).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of points provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_points, unmatched_trks = associate_points_to_trackers(
            points, trks, self.distance_threshold)

        # update matched trackers with assigned points
        for m in matched:
            self.trackers[m[1]].update(points[m[0], :])

        # create and initialise new trackers for unmatched points
        for i in unmatched_points:
            trk = KalmanPointTracker(points[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) \
                    and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 4))


def noise(points, noise_level=0.5):
    """
    Add noise to points
    """
    return points + np.random.randn(*points.shape) * noise_level


def plot(points, ax, title="", colors=[]):
    """
    Plot tracklets
    """
    xyzt = []
    for i in points:
        for j in i:
            xyzt.append([j[0], j[1], j[2], j[3]])

    xyzt = np.array(xyzt).reshape(-1, 4)
    ax.set_title(title)
    for i in xyzt:
        ax.scatter(i[0], i[1], color=colors[int(i[3])])


if __name__ == '__main__':
    from tqdm import tqdm

    # Create tracker
    tracker = Sort(5, 3, 5)

    # Create points
    op = []
    gt = []
    observer = []
    for i in range(100):
        a = np.array([i, 100-i,  0, 1])
        b = np.array([i, 70,  0, 2])
        c = np.array([i, i-30,  0, 3])
        
        if i <= 30:
            gt.append([a.copy(), b.copy()])
        else:
            gt.append([a.copy(), b.copy(), c.copy()])

        a[:3] = noise(a[:3])
        b[:3] = noise(b[:3])
        c[:3] = noise(c[:3])

        if i <= 20:
            observer.append([a, b])
        elif 20 < i <= 30:
            observer.append([b])
        elif 30 < i <= 50:
            observer.append([a, b, c])
        elif 50 < i <= 55:
            observer.append([a, c])
        elif 55 < i <= 60:
            observer.append([a])
        elif 60 < i <= 65:
            observer.append([a, b])
        else:
            observer.append([a, b, c])

    tracklets = []
    # Track points
    for i, frame_points in enumerate(tqdm(observer)):
        # Track points
        tracked_points = tracker.update(np.array(frame_points)[:, :3])
        tracklets.append(tracked_points)

    fig, (gt_plt, observer_plt, track_plt) = plt.subplots(3, 1, figsize=(8, 12))

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']

    plot(gt, gt_plt, title="Ground Truth", colors=colors)

    plot(observer, observer_plt, title="Observer", colors=colors)

    plot(tracklets, track_plt, title="Tracklets", colors=colors)
    # observer.scatter(xy[:,0],xy[:,1], label='observer')

    # fig.show()
    fig.savefig("sort.png")
