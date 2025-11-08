"""Detection modules for nests and motion."""

from beemonitor.detection.nest_detector import NestDetector
from beemonitor.detection.motion_tracking import HyDaTTracker as MotionDetector

__all__ = ["NestDetector", "MotionDetector"]