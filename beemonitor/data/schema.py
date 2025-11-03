from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# Core units are seconds, frames are ints, pixels are float

@dataclass
class VideoMeta:
    video_id: str              # unique name (e.g., file stem)
    path: str                  # absolute or project-relative
    fps: float
    width: int
    height: int
    site: Optional[str] = None
    date_iso: Optional[str] = None  # e.g., "2025-05-12"
    notes: Optional[str] = None

@dataclass
class FrameAnn:                 # per-frame annotations (optional)
    frame: int
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # e.g., ["bee"], ["nest"]

@dataclass
class TrackAnn:
    track_id: int
    frames: List[int]
    bboxes: List[Tuple[float, float, float, float]]
    labels: Optional[List[str]] = None               # class per frame (optional)

@dataclass
class VideoRecord:
    meta: VideoMeta
    frame_anns: List[FrameAnn] = field(default_factory=list)
    track_anns: List[TrackAnn] = field(default_factory=list)
    # optional paths to pipeline products
    tubes_json: Optional[str] = None
    trajectories_json: Optional[str] = None
    events_json: Optional[str] = None
    visits_csv: Optional[str] = None
    summary_csv: Optional[str] = None

@dataclass
class DatasetIndex:
    """Lightweight manifest for a collection of videos."""
    root: str                                  # project root for relative paths
    videos: Dict[str, VideoRecord]             # key = video_id
    splits: Dict[str, List[str]] = field(default_factory=dict)  # e.g. {"train":[...],"val":[...],"test":[...]}

    def split_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}
