from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Direction(str, Enum):
    IN = "in"
    OUT = "out"

@dataclass
class Event:
    tube_id: str
    track_id: int
    direction: Direction          # "in" or "out"
    frame_in: int                 # frame index where crossing occurred
    time_in_s: float              # seconds (frame_in / fps)
    confidence: Optional[float] = None  # optional aggregate conf around the crossing
