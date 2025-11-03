from __future__ import annotations
from typing import Any, Dict
from .aggregator import EventAnalyzer

def build_analyzer(cfg: Dict[str, Any]) -> EventAnalyzer:
    a = (cfg.get("analyzer") or {})
    return EventAnalyzer(
        allow_multiple_open=bool(a.get("allow_multiple_open", False)),
    )
