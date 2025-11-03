from __future__ import annotations
from typing import Any, Dict, Iterable
from .beetrack import BeeTrack
from .multitracker import MultiClassTracker

def build_tracker(cfg: Dict[str, Any]):
    t_cfg = cfg.get("tracking", {}) or {}
    name = (t_cfg.get("name") or "beetrack").lower()

    if name == "beetrack":
        return BeeTrack(
            max_age=int(t_cfg.get("max_age", 15)),
            min_hits=int(t_cfg.get("min_hits", 3)),
            dist_threshold=float(t_cfg.get("dist_threshold", 50)),
        )

    if name in ("multibeetrack", "multi"):
        cls_ids = t_cfg.get("classes", [0])  # e.g., [0,1] for bees & nests
        return MultiClassTracker(
            class_ids=list(map(int, cls_ids)),
            max_age=int(t_cfg.get("max_age", 15)),
            min_hits=int(t_cfg.get("min_hits", 3)),
            dist_threshold=float(t_cfg.get("dist_threshold", 50)),
            global_id_space=bool(t_cfg.get("global_id_space", False)),
        )

    raise ValueError(f"Unknown tracker type: {name}")
