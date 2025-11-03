from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from beemonitor.types import Trajectory, Tube
from .geometry import poly_to_bbox, inflate_bbox, point_in_bbox
from .model import Event, Direction


class EventExtractor:
    """
    Entry/exit detection using per-tube inner/outer gates (hysteresis) plus debouncing.
    Emits IN on outside_outer -> inside_inner, OUT on inside_inner -> outside_outer.
    Also supports flush_on_end: if a track ends outside its tube, force an OUT.
    """

    def __init__(
        self,
        in_inflate_px: float = 4.0,
        out_inflate_px: float = 10.0,
        min_dwell_frames: int = 3,
        min_gap_frames: int = 5,
        max_assign_dist_px: float = 40.0,
        flush_on_end: bool = True,
    ):
        assert out_inflate_px >= in_inflate_px, "outer must be >= inner"
        self.in_inflate_px = float(in_inflate_px)
        self.out_inflate_px = float(out_inflate_px)
        self.min_dwell_frames = int(min_dwell_frames)
        self.min_gap_frames = int(min_gap_frames)
        self.max_assign_dist_px = float(max_assign_dist_px)
        self.flush_on_end = bool(flush_on_end)

    # ---------- public API ----------

    def extract(self, trajectories: List[Trajectory], tubes: List[Tube], fps: float) -> List[Event]:
        tube_gates = self._build_gates(tubes)
        last_event_frame_per_tube: Dict[str, int] = {}
        events: List[Event] = []

        # Precompute tube centers
        tube_centers = {t.tube_id: self._tube_center(t) for t in tubes if t.poly}

        for traj in trajectories:
            centers = [self._bbox_center(bb) for bb in traj.bboxes]
            frames = traj.frames

            current_tube: Optional[str] = None
            dwell_count = 0
            last_tube_seen: Optional[str] = None

            for k, (fidx, c) in enumerate(zip(frames, centers)):
                # Determine candidate tube hits (inner first, outer for hysteresis)
                tube_hit_inner, tube_hit_outer = self._query_gates(c, tube_gates, tube_centers)

                if current_tube is None:
                    # Currently OUTSIDE
                    if tube_hit_inner is not None:
                        if last_tube_seen == tube_hit_inner:
                            dwell_count += 1
                        else:
                            dwell_count = 1
                            last_tube_seen = tube_hit_inner

                        if dwell_count >= self.min_dwell_frames:
                            if self._tube_gap_ok(last_event_frame_per_tube, tube_hit_inner, fidx):
                                events.append(Event(
                                    tube_id=tube_hit_inner,
                                    track_id=traj.track_id,
                                    direction=Direction.IN,
                                    frame_in=fidx,
                                    time_in_s=fidx / fps,
                                    confidence=self._local_confidence(traj, k)
                                ))
                                last_event_frame_per_tube[tube_hit_inner] = fidx
                            current_tube = tube_hit_inner
                            dwell_count = 0
                    else:
                        # remain outside; decay memory
                        if last_tube_seen is not None:
                            dwell_count = max(0, dwell_count - 1)
                            if dwell_count == 0:
                                last_tube_seen = None

                else:
                    # Currently INSIDE current_tube
                    if tube_hit_outer == current_tube:
                        # still inside outer gate → maintain state
                        dwell_count = 0
                    else:
                        # candidate exit
                        dwell_count += 1
                        if dwell_count >= self.min_dwell_frames:
                            if self._tube_gap_ok(last_event_frame_per_tube, current_tube, fidx):
                                events.append(Event(
                                    tube_id=current_tube,
                                    track_id=traj.track_id,
                                    direction=Direction.OUT,
                                    frame_in=fidx,
                                    time_in_s=fidx / fps,
                                    confidence=self._local_confidence(traj, k)
                                ))
                                last_event_frame_per_tube[current_tube] = fidx
                            current_tube = None
                            dwell_count = 0
                            last_tube_seen = None

            # --- flush-on-end: if track ends outside its tube's OUTER gate, force OUT
            if self.flush_on_end and current_tube is not None:
                last_frame = frames[-1]
                last_center = centers[-1]
                inner_outer = tube_gates.get(current_tube, None)
                if inner_outer is not None:
                    _, outer = inner_outer
                    if not point_in_bbox(last_center, outer):
                        if self._tube_gap_ok(last_event_frame_per_tube, current_tube, last_frame):
                            events.append(Event(
                                tube_id=current_tube,
                                track_id=traj.track_id,
                                direction=Direction.OUT,
                                frame_in=last_frame,
                                time_in_s=last_frame / fps,
                                confidence=self._local_confidence(traj, len(frames)-1)
                            ))
                            last_event_frame_per_tube[current_tube] = last_frame

        return events

    # ---------- internals ----------

    def _build_gates(self, tubes: List[Tube]):
        gates = {}
        for t in tubes:
            if not t.poly:
                continue
            base = poly_to_bbox(t.poly)
            inner = inflate_bbox(base, self.in_inflate_px)
            outer = inflate_bbox(base, self.out_inflate_px)
            gates[t.tube_id] = (inner, outer)
        return gates

    def _query_gates(self, c: Tuple[float, float], tube_gates, tube_centers):
        hit_inners, hit_outers = [], []
        for tid, (inner, outer) in tube_gates.items():
            if point_in_bbox(c, inner):
                hit_inners.append(tid)
            if point_in_bbox(c, outer):
                hit_outers.append(tid)

        def nearest(tids):
            if not tids:
                return None
            cx, cy = c
            best, bestd = None, 1e18
            for tid in tids:
                tcx, tcy = tube_centers[tid]
                d = (tcx - cx) ** 2 + (tcy - cy) ** 2
                if d < bestd:
                    bestd, best = d, tid
            if np.sqrt(bestd) > self.max_assign_dist_px:
                return None
            return best

        return nearest(hit_inners), nearest(hit_outers)

    @staticmethod
    def _bbox_center(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _tube_center(t: Tube):
        (x1, y1), (x2, _), (_, y2), _ = t.poly
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _local_confidence(traj: Trajectory, k: int, window: int = 3) -> float:
        if not traj.confs:
            return float("nan")
        a = max(0, k - window)
        b = min(len(traj.confs), k + window + 1)
        vals = [c for c in traj.confs[a:b] if c is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def _tube_gap_ok(self, last_event_frame_per_tube: Dict[str, int], tube_id: str, fidx: int) -> bool:
        last = last_event_frame_per_tube.get(tube_id, None)
        if last is None:
            return True
        return (fidx - last) >= self.min_gap_frames
