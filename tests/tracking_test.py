# tests/tracking_interface_test.py
import numpy as np
from beemonitor.tracking.factory import build_tracker
from beemonitor.types import Detection

def _make_dets_for_single_mover(num_frames=8):
    dets_per_frame = []
    for f in range(num_frames):
        x1 = 10 + 5 * f
        y1 = 10
        x2 = x1 + 10
        y2 = 20
        dets_per_frame.append([Detection(frame=f, bbox=(x1, y1, x2, y2), conf=0.9, cls="bee")])
    return dets_per_frame

def test_centroid_tracker_tracks_one_object():
    cfg = {
        "name": "beetrack",
        "max_age": 3,
        "iou_threshold": 0.1,
        "dist_threshold": 30.0,
    }
    tracker = build_tracker(cfg)
    dets_per_frame = _make_dets_for_single_mover(10)
    trajs = tracker.track_detections(dets_per_frame, fps=30.0)
    assert isinstance(trajs, list)
    assert len(trajs) == 1
    t = trajs[0]
    assert len(t.frames) == 10
    assert t.track_id >= 1
