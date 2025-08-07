# from norfair import Tracker, Detection
# import numpy as np

# def euclidean_distance(detection, tracked_object):
#     return np.linalg.norm(detection.points - tracked_object.estimate)

# class FaceTracker:
#     def __init__(self):
#         self.tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

#     def update(self, faces):
#         detections = []
#         for f in faces:
#             x1, y1, x2, y2 = f.bbox.astype(int)
#             x_center = (x1 + x2) / 2
#             y_center = (y1 + y2) / 2
#             detections.append(Detection(points=np.array([[x_center, y_center]])))
#         return self.tracker.update(detections)