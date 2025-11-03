
from dataclasses import dataclass
from typing import List, Tuple, Optional

BBox = Tuple[float, float, float, float]      # x1,y1,x2,y2
Point = Tuple[float, float] # middle point of nest x,y

@dataclass
class Detection:
    frame: int
    bbox: BBox
    conf: float
    cls: int
    track_id: Optional[int] = None  # optional, for tracking

@dataclass
class Trajectory:
    track_id: int
    frames: List[int]
    bboxes: List[BBox]
    confs: Optional[List[float]] = None

@dataclass
class Tube:
    tube_id: str
    poly: List[Point]
