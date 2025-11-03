from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from beemonitor.events.model import Direction

@dataclass
class Visit:
    tube_id: str
    track_id: int
    in_frame: int
    out_frame: Optional[int]
    in_time_s: float
    out_time_s: Optional[float]
    dwell_s: Optional[float]
    class_id: Optional[int] = None
    species_name: Optional[str] = None

@dataclass
class TubeSummary:
    tube_id: str
    n_visits: int
    total_dwell_s: float
    mean_dwell_s: float
    class_id: Optional[int] = None
    species_name: Optional[str] = None

