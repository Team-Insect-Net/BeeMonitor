from __future__ import annotations
from typing import Any, Dict
from .extractor import EventExtractor

def build_event_extractor(cfg: Dict[str, Any]) -> EventExtractor:
    """
    YAML example:
    events:
      gate:
        in_inflate_px: 4
        out_inflate_px: 10
        min_dwell_frames: 3
        min_gap_frames: 5
        max_assign_dist_px: 40
        flush_on_end: true
    """
    g = (cfg.get("events") or {}).get("gate", {})
    return EventExtractor(
        in_inflate_px=float(g.get("in_inflate_px", 4.0)),
        out_inflate_px=float(g.get("out_inflate_px", 10.0)),
        min_dwell_frames=int(g.get("min_dwell_frames", 3)),
        min_gap_frames=int(g.get("min_gap_frames", 5)),
        max_assign_dist_px=float(g.get("max_assign_dist_px", 40.0)),
        flush_on_end=bool(g.get("flush_on_end", True)),
    )
