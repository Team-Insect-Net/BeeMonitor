from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from beemonitor.types import Tube

class NestIdentifier(ABC):
    """Interface for nest/tube detection & ID assignment."""

    @abstractmethod
    def detect_tubes(self, frame_bgr: np.ndarray) -> List[Tube]:
        """Detect and return tube polygons/boxes with consistent IDs."""
        raise NotImplementedError

    @abstractmethod
    def detect_or_load(self, video_path: str) -> List[Tube]:
        """Load cached mapping for a video or compute it from the first frame."""
        raise NotImplementedError
