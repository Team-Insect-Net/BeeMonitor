from __future__ import annotations
from ast import Tuple
from collections import defaultdict
from typing import Dict, List, Optional, Iterable
import statistics

from beemonitor.events.model import Event, Direction
from .model import Visit, TubeSummary


class EventAnalyzer:
    """
    Turns raw IN/OUT events into visits and per-tube summaries.

    Pairing rule:
      - For each tube_id, sort events by time.
      - Whenever an IN appears and there's no open visit, open it.
      - OUT closes the most-recent open visit (FIFO=stack of size ≤1 per tube).
      - Multiple INs without OUT: the newest IN supersedes the previous (old one is treated as abandoned)
        unless 'allow_multiple_open' is set (default False).
    """

    def __init__(self, allow_multiple_open: bool = False):
        self.allow_multiple_open = bool(allow_multiple_open)

    def build_visits(self, events: List[Event]) -> List[Visit]:
        visits: List[Visit] = []
        by_tube: Dict[str, List[Event]] = defaultdict(list)
        for e in events:
            by_tube[e.tube_id].append(e)

        for tube_id, evs in by_tube.items():
            evs.sort(key=lambda e: e.time_in_s)
            open_evt: Optional[Event] = None
            for e in evs:
                if e.direction == Direction.IN:
                    open_evt = e
                elif e.direction == Direction.OUT and open_evt:
                    dwell = e.time_in_s - open_evt.time_in_s
                    v = Visit(
                        tube_id=tube_id,
                        track_id=e.track_id,
                        in_frame=open_evt.frame_in,
                        out_frame=e.frame_in,
                        in_time_s=open_evt.time_in_s,
                        out_time_s=e.time_in_s,
                        dwell_s=dwell,
                        class_id=getattr(e, "class_id", None),
                        species_name=getattr(e, "species_name", None),
                    )
                    visits.append(v)
                    open_evt = None
        return visits


    def summarize(self, visits: List[Visit]) -> List[TubeSummary]:
        by_key: Dict[Tuple[str, Optional[int]], List[Visit]] = defaultdict(list)
        for v in visits:
            by_key[(v.tube_id, v.class_id)].append(v)

        summaries: List[TubeSummary] = []
        for (tube_id, cid), vs in by_key.items():
            total_dwell = sum(v.dwell_s or 0.0 for v in vs)
            mean_dwell = total_dwell / len(vs) if vs else 0.0
            species_name = vs[0].species_name if vs else None
            summaries.append(TubeSummary(
                tube_id=tube_id,
                n_visits=len(vs),
                total_dwell_s=total_dwell,
                mean_dwell_s=mean_dwell,
                class_id=cid,
                species_name=species_name,
            ))
        return summaries

