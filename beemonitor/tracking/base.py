from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional
import numpy as np

from beemonitor.types import Detection, Trajectory


class Tracker(ABC):
    """
    Abstract tracker. Implementations may support either:
      - track_detections: track from pre-computed detections per frame
      - track_video: track directly from a video (e.g., via legacy pipeline)
    """

    # --- Primary, detector-agnostic path ---
    def track_detections(
        self,
        dets_per_frame: List[List[Detection]],
        fps: float,
    ) -> List[Trajectory]:
        """Default: not implemented. Implement in subclass if supported."""
        raise NotImplementedError("track_detections not implemented for this tracker.")

    # --- Optional, video path (for legacy adapters) ---
    def track_video(
        self,
        video_path: str,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[Trajectory]:
        """Default: not implemented. Implement in subclass if supported."""
        raise NotImplementedError("track_video not implemented for this tracker.")
