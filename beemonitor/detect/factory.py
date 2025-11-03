from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .base import Detector
from .yolo import YoloDetector

@dataclass
class DetectorBundle:
    """Container for multiple named detectors (e.g., bee + nest)."""
    bee: Optional[Detector] = None
    nest: Optional[Detector] = None
    # you can add more named detectors here if needed

def build_detector(cfg: Dict[str, Any]):
    name = (cfg.get("name") or "yolo").lower()
    if name != "yolo":
        raise ValueError(f"Unknown detector: {name}")

    conf = cfg.get("conf", cfg.get("conf_threshold", 0.25))
    iou  = cfg.get("iou",  cfg.get("iou_threshold", 0.45))
    return YoloDetector(
        weights=cfg["weights"],
        conf=float(conf),
        iou=float(iou),
        classes=cfg.get("classes", None),        # post-filtered
        device=cfg.get("device", None),
        imgsz=cfg.get("imgsz", None),
        verbose=bool(cfg.get("verbose", False)),
        apply_classes_filter_post=True,          # ← safer
    )

def build_detectors(cfg_map: Mapping[str, Dict[str, Any]]) -> DetectorBundle:
    """
    Build multiple named detectors from a map, e.g.:

    cfg_map = {
      "bee": {...},   # YOLO weights for bee model
      "nest": {...},  # YOLO weights for nest/tube model
    }
    """
    bundle = DetectorBundle()
    for name, subcfg in cfg_map.items():
        d = build_detector(subcfg)
        if name.lower() == "bee":
            bundle.bee = d
        elif name.lower() == "nest":
            bundle.nest = d
        else:
            # If you want extra named detectors, extend DetectorBundle and add branches here.
            setattr(bundle, name, d)  # store dynamically
    return bundle
