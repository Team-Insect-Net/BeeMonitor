from __future__ import annotations
from typing import List, Any

from beemonitor.types import Trajectory, BBox
from .base import Tracker


def _to_trajectory_list(wrapper_tracks: Any) -> List[Trajectory]:
    """
    Convert legacy wrapper outputs into List[Trajectory].

    Expected minimal shape (examples):
      - List[dict]: {"id": int, "frames": [...], "bboxes": [[x1,y1,x2,y2], ...], "confs": [...]}
      - Or List[Tuple[track_id, frames, bboxes, confs]]
    Adjust if your wrapper uses a different structure.
    """
    out: List[Trajectory] = []
    for t in wrapper_tracks or []:
        if isinstance(t, dict):
            out.append(
                Trajectory(
                    track_id=int(t.get("id", t.get("track_id", -1))),
                    frames=[int(x) for x in t.get("frames", [])],
                    bboxes=[tuple(map(float, bb)) for bb in t.get("bboxes", [])],  # type: ignore
                    confs=[float(c) for c in t.get("confs", [])] if t.get("confs") is not None else None,
                )
            )
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            track_id = int(t[0])
            frames = [int(x) for x in (t[1] or [])]
            bboxes = [tuple(map(float, bb)) for bb in (t[2] or [])]  # type: ignore
            confs = [float(c) for c in (t[3] or [])] if len(t) > 3 and t[3] is not None else None
            out.append(Trajectory(track_id=track_id, frames=frames, bboxes=bboxes, confs=confs))
        else:
            # As a last resort, skip unknown shapes
            continue
    return out


class UltralyticsByteTrack(Tracker):
    """
    Adapter over your legacy Ultralytics_Tracker_wrapper.

    NOTE: This implementation uses the video path entrypoint; if your wrapper
    exposes a function that works from detections per frame, you can also
    implement `track_detections` similarly.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path

    def track_video(self, video_path: str, fps: float | None = None, **kwargs) -> List[Trajectory]:
        try:
            # Import your legacy wrapper dynamically
            from beemonitor.legacy import Ultralytics_Tracker_wrapper as legacy
        except Exception as e:
            raise RuntimeError(
                "Could not import legacy Ultralytics_Tracker_wrapper. Ensure it exists under "
                "`beemonitor/legacy/Ultralytics_Tracker_wrapper.py`."
            ) from e

        # You may need to adjust the call signature to match your wrapper.
        # Example expected call (replace with your actual function/method names):
        # tracks = legacy.get_tracks(video_path=video_path, cfg=self.config_path, **kwargs)
        if hasattr(legacy, "get_tracks"):
            tracks = legacy.get_tracks(video_path=video_path, cfg=self.config_path, **kwargs)
        elif hasattr(legacy, "getTracks"):
            tracks = legacy.getTracks(video_path=video_path, cfg=self.config_path, **kwargs)
        else:
            raise RuntimeError(
                "Legacy wrapper missing a `get_tracks`/`getTracks` function. Please expose one."
            )

        return _to_trajectory_list(tracks)
