# beemonitor/tracking/beetrack.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import Counter
import math

from beemonitor.types import Detection  # expects .bbox=(x1,y1,x2,y2), .conf: float, .cls:int
from .model import Trajectory


def _centroid(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


@dataclass
class _Tracklet:
    track_id: int
    last_center: Tuple[float, float]
    last_bbox: Tuple[float, float, float, float]
    age: int = 0                 # frames since last match
    hits: int = 1                # number of matches
    frames: List[int] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)  # ← new: per-class votes

    def add(self, fidx: int, det: Detection):
        self.last_bbox = det.bbox
        self.last_center = _centroid(det.bbox)
        self.frames.append(fidx)
        self.bboxes.append(det.bbox)
        self.confs.append(float(getattr(det, "conf", 1.0)))
        cls_val = getattr(det, "cls", None)
        if cls_val is not None:
            try:
                self.class_counts[int(cls_val)] += 1
            except Exception:
                pass
        self.hits += 1
        self.age = 0  # reset on match


class BeeTrack:
    """
    Lightweight centroid tracker tuned for bees.
    - Greedy nearest-neighbor assignment with distance threshold
    - Tracks age out after `max_age` missed frames
    - Only yields trajectories with >= min_hits
    - NEW: assigns majority-vote class_id to each finished trajectory
    """

    def __init__(self, max_age: int = 15, min_hits: int = 3, dist_threshold: float = 50.0):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.dist_threshold = float(dist_threshold)

        self._next_id = 1
        self._tracks: List[_Tracklet] = []
        self._finished: List[Trajectory] = []

    def update(self, frame_idx: int, detections: List[Detection]) -> None:
        """Ingest detections for a frame and update internal tracks."""
        det_centers = [_centroid(d.bbox) for d in detections]
        det_used = [False] * len(detections)

        # match existing tracks
        for tr in self._tracks:
            best_j = -1
            best_d = float("inf")
            for j, (c, used) in enumerate(zip(det_centers, det_used)):
                if used:
                    continue
                d = _euclid(tr.last_center, c)
                if d < best_d:
                    best_d = d
                    best_j = j

            if best_j >= 0 and best_d <= self.dist_threshold:
                tr.add(frame_idx, detections[best_j])
                det_used[best_j] = True
            else:
                tr.age += 1

        # spawn new tracks for unmatched detections
        for j, used in enumerate(det_used):
            if not used:
                det = detections[j]
                tr = _Tracklet(
                    track_id=self._next_id,
                    last_center=_centroid(det.bbox),
                    last_bbox=det.bbox,
                    age=0,
                    hits=1,
                    frames=[frame_idx],
                    bboxes=[det.bbox],
                    confs=[float(getattr(det, "conf", 1.0))],
                )
                cls_val = getattr(det, "cls", None)
                if cls_val is not None:
                    try:
                        tr.class_counts[int(cls_val)] += 1
                    except Exception:
                        pass
                self._tracks.append(tr)
                self._next_id += 1

        # retire aged-out tracks
        self._retire_old_tracks()

    def finalize(self) -> List[Trajectory]:
        """Finish all remaining active tracks and return all trajectories."""
        for tr in self._tracks:
            self._maybe_finish(tr)
        self._tracks.clear()
        out = self._finished
        self._finished = []
        return out

    def _retire_old_tracks(self):
        still_active: List[_Tracklet] = []
        for tr in self._tracks:
            if tr.age > self.max_age:
                self._maybe_finish(tr)
            else:
                still_active.append(tr)
        self._tracks = still_active

    def _maybe_finish(self, tr: _Tracklet):
        if tr.hits >= self.min_hits and len(tr.frames) >= self.min_hits:
            class_id: Optional[int] = None
            if tr.class_counts:
                # majority vote
                class_id = max(tr.class_counts.items(), key=lambda kv: kv[1])[0]

            self._finished.append(Trajectory(
                track_id=tr.track_id,
                frames=tr.frames,
                bboxes=tr.bboxes,
                confs=tr.confs,
                class_id=class_id,   # ← majority class assigned here
            ))
