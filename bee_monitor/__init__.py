"""
Bee Monitor: Computer vision system for monitoring solitary bee activity.

This package provides tools for detecting, tracking, and analyzing bee behavior
in bee hotel videos using YOLO-based object detection and custom tracking algorithms.
"""

__version__ = "1.0.0"
__author__ = "Edward Amoah"
__email__ = "your.email@example.com"

from bee_monitor.core.video_analyzer import BeeMonitor, AnalysisResults
from bee_monitor.core.config import Config

__all__ = [
    "BeeMonitor",
    "AnalysisResults",
    "Config",
]