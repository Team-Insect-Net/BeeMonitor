from __future__ import annotations
import os, json, random
from typing import Dict, List, Tuple
from .schema import DatasetIndex, VideoMeta, VideoRecord

def save_index(idx: DatasetIndex, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _todict(v):
        if hasattr(v, "__dict__"):
            d = v.__dict__.copy()
            for k, val in d.items():
                if isinstance(val, list):
                    d[k] = [ _todict(x) for x in val ]
                elif hasattr(val, "__dict__"):
                    d[k] = _todict(val)
            return d
        return v
    with open(path, "w") as f:
        json.dump(_todict(idx), f, indent=2)

def load_index(path: str) -> DatasetIndex:
    from .schema import FrameAnn, TrackAnn
    d = json.load(open(path))
    vids: Dict[str, VideoRecord] = {}
    for vid, rec in d["videos"].items():
        meta = VideoMeta(**rec["meta"])
        frame_anns = [FrameAnn(**fa) for fa in rec.get("frame_anns", [])]
        track_anns = [TrackAnn(**ta) for ta in rec.get("track_anns", [])]
        vids[vid] = VideoRecord(
            meta=meta, frame_anns=frame_anns, track_anns=track_anns,
            tubes_json=rec.get("tubes_json"), trajectories_json=rec.get("trajectories_json"),
            events_json=rec.get("events_json"), visits_csv=rec.get("visits_csv"),
            summary_csv=rec.get("summary_csv")
        )
    return DatasetIndex(root=d["root"], videos=vids, splits=d.get("splits", {}))

def add_video(idx: DatasetIndex, meta: VideoMeta) -> None:
    if meta.video_id in idx.videos:
        raise ValueError(f"video_id exists: {meta.video_id}")
    idx.videos[meta.video_id] = VideoRecord(meta=meta)

def make_random_splits(idx: DatasetIndex, train=0.7, val=0.15, seed=42) -> None:
    ids = list(idx.videos.keys())
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n*train)
    n_val = int(n*val)
    idx.splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train+n_val],
        "test": ids[n_train+n_val:]
    }
