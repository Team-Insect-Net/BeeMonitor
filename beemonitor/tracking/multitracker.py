from __future__ import annotations
from typing import Dict, List, Optional, Iterable
from collections import defaultdict

from beemonitor.types import Detection
from .beetrack import BeeTrack
from .model import Trajectory


class MultiClassTracker:
    """
    Wraps N BeeTrack trackers, one per class_id.
    - Efficient: you call YOLO once, pass all detections here.
    - Safe: IDs are unique per class by default (optionally add class prefix).
    """

    def __init__(
        self,
        class_ids: Iterable[int],
        max_age: int = 15,
        min_hits: int = 3,
        dist_threshold: float = 50.0,
        global_id_space: bool = False,
    ):
        self._per_class: Dict[int, BeeTrack] = {
            int(cid): BeeTrack(max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold)
            for cid in class_ids
        }
        self.global_id_space = bool(global_id_space)
        self._id_offset: Dict[int, int] = {int(cid): (cid * 10_000 if global_id_space else 0) for cid in class_ids}

    def update(self, frame_idx: int, detections: List[Detection]) -> None:
        by_cls: Dict[int, List[Detection]] = defaultdict(list)
        for d in detections:
            # Only route detections that have a class in our set
            cid = int(getattr(d, "cls", -1))
            if cid in self._per_class:
                by_cls[cid].append(d)
        for cid, trk in self._per_class.items():
            trk.update(frame_idx, by_cls.get(cid, []))

    def finalize(self) -> List[Trajectory]:
        out: List[Trajectory] = []
        for cid, trk in self._per_class.items():
            for t in trk.finalize():
                # annotate class_id; optionally offset IDs into a global space
                t.class_id = cid
                if self.global_id_space and t.track_id is not None:
                    t.track_id = self._id_offset[cid] + int(t.track_id)
                out.append(t)
        return out
