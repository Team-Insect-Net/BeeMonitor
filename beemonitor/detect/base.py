from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List
import numpy as np

from beemonitor.types import Detection

class Detector(ABC):
    """Abstract detector interface."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame and return a list of Detection."""
        raise NotImplementedError

    def batch_predict(self, frames_bgr: Iterable[np.ndarray]) -> Iterable[List[Detection]]:
        """Default slow batch predict that just loops over frames."""
        for f in frames_bgr:
            yield self.predict(f)
