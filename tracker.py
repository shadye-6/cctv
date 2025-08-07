from ultralytics.trackers.bot_sort import BoTSORT
from ultralytics.utils.plotting import colors
import numpy as np

class FaceTracker:
    def __init__(self):
        self.tracker = BoTSORT(
            reid=False,
            track_high_thresh=0.3,
            track_low_thresh=0.05,
            new_track_thresh=0.7,
            track_buffer=30
        )

    def update(self, faces, frame):
        detections = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = 1.0  # InsightFace doesn't return confidence, assume high
            detections.append([x1, y1, x2, y2, conf])

        if not detections:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets, frame)

        return tracks  # Format: [x1, y1, x2, y2, id]
