from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import os
import cv2
import numpy as np

from beemonitor.events.model import Event, Direction
from beemonitor.tracking.model import Trajectory

BBox = Tuple[float, float, float, float]

# ---- simple colors (BGR) ----
C_TUBE   = (80, 255, 80)      # green
C_TEXT   = (230, 230, 230)    # light grey
C_TRACK  = (255, 160, 40)     # orange
C_TAIL   = (120, 200, 255)    # light blue
C_EVENT_IN  = (60, 220, 255)  # yellow-ish
C_EVENT_OUT = (60, 60, 255)   # red


@dataclass
class Tube:
    tube_id: str
    bbox: BBox


def _put_label(img, text, org, scale=0.6, color=C_TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _draw_tube(img, tube: Tube):
    x1, y1, x2, y2 = map(int, tube.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), C_TUBE, 2)
    _put_label(img, str(tube.tube_id), (x1, max(0, y1 - 6)), scale=0.5, color=C_TUBE, thickness=1)


def _draw_traj_state(img, traj: Trajectory, frame_idx: int, tail_len: int = 20):
    """
    Draw current bbox + short tail of centroids ending at frame_idx.
    """
    # pick last bbox at or before frame_idx
    if not traj.frames:
        return
    # find index of last frame <= frame_idx
    import bisect
    k = bisect.bisect_right(traj.frames, frame_idx) - 1
    if k < 0:
        return

    # current bbox
    x1, y1, x2, y2 = map(int, traj.bboxes[k])
    cv2.rectangle(img, (x1, y1), (x2, y2), C_TRACK, 2)

    # id/species label
    label = f"ID {traj.track_id}"
    if getattr(traj, "species_name", None):
        label += f" • {traj.species_name}"
    elif getattr(traj, "class_id", None) is not None:
        label += f" • c{traj.class_id}"
    _put_label(img, label, (x1, min(img.shape[0]-4, y2 + 14)), scale=0.55, color=C_TRACK)

    # tail (centers of last N boxes up to k)
    xs, ys = [], []
    s = max(0, k - tail_len + 1)
    for j in range(s, k + 1):
        bx1, by1, bx2, by2 = traj.bboxes[j]
        xs.append(0.5 * (bx1 + bx2))
        ys.append(0.5 * (by1 + by2))
    pts = np.column_stack([xs, ys]).astype(int)
    for i in range(1, len(pts)):
        cv2.line(img, (pts[i-1,0], pts[i-1,1]), (pts[i,0], pts[i,1]), C_TAIL, 2, lineType=cv2.LINE_AA)


def _index_events_by_frame(events: List[Event]) -> Dict[int, List[Event]]:
    by_f: Dict[int, List[Event]] = {}
    for e in events:
        f = int(getattr(e, "frame_in", 0))
        by_f.setdefault(f, []).append(e)
    return by_f


def _draw_events(img, evs: Iterable[Event], tubes_by_id: Dict[str, Tube]):
    """
    Draw a small marker near the tube box for each event at this frame.
    IN: filled circle; OUT: cross.
    """
    for e in evs:
        t = tubes_by_id.get(str(e.tube_id))
        if not t:
            continue
        x1, y1, x2, y2 = map(int, t.bbox)
        cx = (x1 + x2) // 2
        cy = y1 - 10  # just above the tube
        if e.direction == Direction.IN:
            cv2.circle(img, (cx, cy), 5, C_EVENT_IN, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (cx, cy), C_EVENT_OUT, markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)


def render_overlay_video(
    video_path: str,
    tubes: List[Tube],
    trajectories: List[Trajectory],
    events: List[Event],
    fps: float,
    out_path: str,
    draw_tubes: bool = True,
    draw_tracks: bool = True,
    tail_len: int = 20,
    every_nth_frame: int = 1,
    fourcc: str = "mp4v",
) -> str:
    """
    Render an annotated video with nest IDs, trajectories, and IN/OUT events.

    Returns
    -------
    out_path : str
        Path to the written video.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps / max(1, every_nth_frame), (W, H))

    tubes_by_id = {str(t.tube_id): t for t in tubes}
    ev_by_frame = _index_events_by_frame(events)

    fidx = 0
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx % every_nth_frame) != 0:
            fidx += 1
            continue

        # draw…
        if draw_tubes:
            for t in tubes:
                _draw_tube(frame, t)

        if draw_tracks:
            # draw current bbox + short tail for all trajectories that have started
            for tr in trajectories:
                _draw_traj_state(frame, tr, fidx, tail_len=tail_len)

        if fidx in ev_by_frame:
            _draw_events(frame, ev_by_frame[fidx], tubes_by_id)

        # small HUD
        _put_label(frame, f"BeeMonitor Overlay • frame {fidx}/{total}", (10, H-10), scale=0.5, color=(180,180,180))

        out.write(frame)
        fidx += 1

    out.release()
    cap.release()
    return out_path
