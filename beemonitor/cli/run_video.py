# beemonitor/cli/run_video.py
from __future__ import annotations

import os
import cv2
import json
import logging
from typing import Dict, List, Optional

from beemonitor.detect.factory import build_detector
from beemonitor.tracking.factory import build_tracker
from beemonitor.types import Detection

# Try to import your canonical frame iterator
try:
    from beemonitor.utils.video import iter_frames
except Exception:
    # Minimal fallback iterator
    def iter_frames(video_path: str):
        cap = cv2.VideoCapture(video_path)
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield fidx, frame
            fidx += 1
        cap.release()


log = logging.getLogger("beemonitor.run_video")


def _setup_logging():
    # Respect env var; default INFO
    level = os.getenv("BEEMONITOR_LOGLEVEL", "INFO").upper()
    level_num = getattr(logging, level, logging.INFO)
    logging.basicConfig(
        level=level_num,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _count_by_class(dets: List[Detection]) -> Dict[int, int]:
    hist: Dict[int, int] = {}
    for d in dets:
        cid = int(getattr(d, "cls", -1))
        hist[cid] = hist.get(cid, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))


def _draw_debug(
    frame,
    dets: List[Detection],
    out_dir: str,
    fidx: int,
    prefix: str = "raw",
):
    """Write a quick debug image with bboxes and class labels."""
    os.makedirs(out_dir, exist_ok=True)
    vis = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox)
        cls_id = int(getattr(d, "cls", -1))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"c{cls_id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_{fidx:06d}.jpg"), vis)


def run_tracker_on_video(cfg: dict, video_path: str):
    """
    End-to-end runner:
      - builds detector & tracker
      - logs per-frame detection histograms by class (before/after filtering)
      - updates tracker
      - returns trajectories and fps

    Returns
    -------
    trajectories : List[Trajectory]
    fps : float
    """
    _setup_logging()

    # ----------------------------------------
    # Build components
    # ----------------------------------------
    det_cfg = cfg.get("detectors", {}).get("bee", {})
    det = build_detector(det_cfg)

    tracker = build_tracker(cfg)  # BeeTrack or MultiClassTracker

    # Which classes to track (e.g., [3], or [0,1,2])
    allowed = set(cfg.get("tracking", {}).get("classes", []) or [])

    # Optional: debug image dump for first N frames
    out_cfg = cfg.get("output", {}) or {}
    debug_dump_dir = out_cfg.get("debug_dump_dir", "outputs/debug")
    debug_dump_first_n = int(out_cfg.get("debug_dump_first_n", 0))  # 0 disables

    # ----------------------------------------
    # FPS and first-frame smoke test
    # ----------------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise RuntimeError(f"[run] Failed to read video: {video_path}")

    log.info(f"[run] video='{video_path}' fps={fps:.2f} size={first.shape[1]}x{first.shape[0]}")
    log.info(f"[run] tracking classes = {sorted(allowed) if allowed else 'ALL'}")
    log.info(f"[run] debug dump: first_n={debug_dump_first_n} → {debug_dump_dir}")

    # ----------------------------------------
    # Frame loop
    # ----------------------------------------
    n_frames = 0
    n_det_total = 0
    n_det_kept = 0
    global_hist_raw: Dict[int, int] = {}
    global_hist_kept: Dict[int, int] = {}

    for fidx, frame in iter_frames(video_path):
        n_frames += 1

        # Detection (single YOLO pass)
        dets: List[Detection] = det.predict(frame)

        print(f"[debug] frame {fidx} got {len(dets)} detections")
        print(dets)

        # Per-frame histogram BEFORE filter
        hist_raw = _count_by_class(dets)
        n_det_total += sum(hist_raw.values())
        for k, v in hist_raw.items():
            global_hist_raw[k] = global_hist_raw.get(k, 0) + v

        # Optional debug dump (raw)
        if debug_dump_first_n and fidx < debug_dump_first_n:
            _draw_debug(frame, dets, debug_dump_dir, fidx, prefix="raw")

        # Filter by allowed tracking classes (if specified)
        if allowed:
            dets = [d for d in dets if int(getattr(d, "cls", -1)) in allowed]

        # Per-frame histogram AFTER filter
        hist_kept = _count_by_class(dets)
        n_det_kept += sum(hist_kept.values())
        for k, v in hist_kept.items():
            global_hist_kept[k] = global_hist_kept.get(k, 0) + v

        # Optional debug dump (kept)
        if debug_dump_first_n and fidx < debug_dump_first_n:
            _draw_debug(frame, dets, debug_dump_dir, fidx, prefix="kept")

        # Per-frame log
        log.info(
            f"[frame {fidx:06d}] raw={sum(hist_raw.values())} {hist_raw}  "
            f"kept={sum(hist_kept.values())} {hist_kept}"
        )

        print(f"[debug] frame {fidx} tracking {len(dets)} detections")
        print(dets)

        # Update tracker
        tracker.update(fidx, dets)

    # ----------------------------------------
    # Finalize + summaries
    # ----------------------------------------
    trajectories = tracker.finalize()
    log.info(
        "[done] frames=%d det_total=%d det_kept=%d "
        "hist_raw=%s hist_kept=%s trajectories=%d",
        n_frames, n_det_total, n_det_kept, json.dumps(global_hist_raw), json.dumps(global_hist_kept),
        len(trajectories),
    )

    # Optional: annotate species name if provided in cfg
    class_names = {
        int(k): v for k, v in (cfg.get("labels", {}).get("class_names", {}) or {}).items()
    }
    for t in trajectories:
        cid = getattr(t, "class_id", None)
        if cid is not None and cid in class_names:
            t.species_name = class_names[cid]

    # Per-class trajectory counts (if class_id available)
    traj_hist: Dict[int, int] = {}
    for t in trajectories:
        cid = int(getattr(t, "class_id", -1))
        traj_hist[cid] = traj_hist.get(cid, 0) + 1
    log.info(f"[done] trajectories_by_class={traj_hist}")

    return trajectories, float(fps)
