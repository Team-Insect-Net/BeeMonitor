# beemonitor/tracking/model.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class Trajectory:
    """
    A single bee trajectory produced by the tracker.

    Attributes
    ----------
    track_id : int
        Stable ID assigned by the tracker.
    frames : List[int]
        Frame indices where this track was observed (monotonic).
    bboxes : List[BBox]
        Bounding boxes aligned with `frames`.
    confs : Optional[List[float]]
        Optional detection confidences aligned with `frames`.
    """
    track_id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    confs: Optional[List[float]] = None
    class_id: Optional[int] = None   
    species_name: Optional[str] = None # human-readable label (filled later)

    # ---------- convenience properties ----------
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def first_frame(self) -> Optional[int]:
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self) -> Optional[int]:
        return self.frames[-1] if self.frames else None

    def duration_frames(self) -> int:
        """Inclusive span (last-first+1) if available, else 0."""
        if not self.frames:
            return 0
        return (self.frames[-1] - self.frames[0] + 1)

    def duration_seconds(self, fps: float) -> float:
        """Approx duration in seconds from first→last frame (inclusive span / fps)."""
        if fps <= 0:
            return 0.0
        return self.duration_frames() / float(fps)

    def centers(self) -> List[Tuple[float, float]]:
        """Sequence of (cx, cy) centers for each bbox."""
        out: List[Tuple[float, float]] = []
        for (x1, y1, x2, y2) in self.bboxes:
            out.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        return out

    def to_dict(self) -> dict:
        """Stable dict for JSON export."""
        d = asdict(self)
        return d


__all__ = ["Trajectory", "BBox"]
