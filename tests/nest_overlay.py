#!/usr/bin/env python3
from __future__ import annotations

import os, json, yaml, cv2, numpy as np
from typing import Any, Dict, List, Tuple

# Your factory (uses beemonitor/config/pipeline.default.yaml)
from beemonitor.nestid.factory import build_nest_identifier

# ---------- helpers ----------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def grab_frame(video_path: str, frame_index: int) -> Tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_index = max(0, min(frame_index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return frame, float(fps), total

def put_text(img, txt, org, scale=0.5, color=(220,220,220), thick=1):
    cv2.putText(img, str(txt), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_box(img, xyxy, color=(80,255,80), thick=2):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)

def draw_cross(img, pt, color=(255,180,60), r=6, thick=2):
    x,y = map(int, pt)
    cv2.line(img, (x-r,y), (x+r,y), color, thick, cv2.LINE_AA)
    cv2.line(img, (x,y-r), (x,y+r), color, thick, cv2.LINE_AA)

# ---------- main debug function ----------

def debug_nest_id(
    cfg_path: str,
    video_path: str,
    out_png: str,
    frame_index: int = 0,
    show_template_centers: bool = True,
    draw_guides: bool = True,
) -> Dict[str, Any]:
    """
    Runs the nest identifier on a single frame and writes a diagnostic PNG.

    Returns a dict with counts + alignment info for quick inspection.
    """
    cfg = read_cfg(cfg_path)
    nid = build_nest_identifier(cfg)

    # 1) read a single frame (we don't need to run YOLO on whole video here)
    frame, fps, total = grab_frame(video_path, frame_index)
    vis = frame.copy()

    # 2) run the identifier in "single-frame" mode if available,
    #    else call the usual detect_or_load(video) and use its cached geometry.
    #    We try detect_or_load first to stay consistent with your pipeline.
    tubes = nid.detect_or_load(video_path)
    # If your impl has a per-frame refinement, you could call it here.

    # tubes should be a list with id + bbox; adapt field access:
    def tube_to_xyxy(t) -> Tuple[float,float,float,float]:
        if isinstance(t, dict):
            if "bbox" in t and t["bbox"] is not None:
                return tuple(map(float, t["bbox"]))  # x1,y1,x2,y2
            for key in ("poly", "points", "polygon"):
                if key in t and t[key]:
                    arr = np.asarray(t[key], dtype=float)
                    xs = arr[..., 0]
                    ys = arr[..., 1]
                    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            if all(k in t for k in ("x1", "y1", "x2", "y2")):
                return float(t["x1"]), float(t["y1"]), float(t["x2"]), float(t["y2"])
            raise ValueError("Tube dict missing bbox/points fields")
        # object
        if hasattr(t, "bbox") and t.bbox is not None:
            return tuple(map(float, t.bbox))
        # many NestID implementations expose polygons instead of raw boxes
        pts = None
        if hasattr(t, "poly"):
            pts = getattr(t, "poly")
        elif hasattr(t, "points"):
            pts = getattr(t, "points")
        if pts is None and isinstance(getattr(t, "__dict__", None), dict):
            data = t.__dict__
            for key in ("poly", "points", "polygon"):
                if key in data:
                    pts = data[key]
                    break
        if pts is not None:
            arr = np.asarray(pts, dtype=float)
            if arr.size == 0 or arr.shape[-1] < 2:
                raise ValueError("Tube polygon missing coordinate pairs")
            xs = arr[..., 0]
            ys = arr[..., 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        return float(t.x1), float(t.y1), float(t.x2), float(t.y2)

    def tube_id(t) -> str:
        return str(t.get("id") if isinstance(t, dict) else getattr(t, "id", ""))

    # 3) draw detected tube boxes + numeric IDs
    for t in tubes:
        xyxy = tube_to_xyxy(t)
        draw_box(vis, xyxy, (80,255,80), 2)
        x1,y1,_,_ = map(int, xyxy)
        put_text(vis, tube_id(t), (x1, max(12, y1-6)), scale=0.5, color=(80,255,80), thick=1)

    # 4) ask the identifier for debug info if it exposes it (alignment, residuals, centers)
    # We try common names; ignore if not present
    # Expecting nid.debug or nid.last_debug dict
    debug_info = {}
    for attr in ("debug", "last_debug", "state", "_debug"):
        if hasattr(nid, attr):
            val = getattr(nid, attr)
            if isinstance(val, dict):
                debug_info = val
                break

    # Draw template centers projected into the image (if provided)
    # Expected keys (optional): 'tpl_centers_px' or 'proj_centers', shape N×2
    centers_px = None
    for k in ("tpl_centers_px", "proj_centers", "template_centers_px"):
        if isinstance(debug_info.get(k), np.ndarray):
            centers_px = debug_info[k]
            break
        if isinstance(debug_info.get(k), list) and len(debug_info[k]) and isinstance(debug_info[k][0], (list, tuple)):
            centers_px = np.asarray(debug_info[k], dtype=float)
            break

    if show_template_centers and centers_px is not None:
        for (cx, cy) in centers_px:
            draw_cross(vis, (cx, cy), color=(255,180,60), r=5, thick=2)

    # Optionally draw rough guides (vertical/horizontal midlines) to check skew
    if draw_guides:
        h, w = vis.shape[:2]
        cv2.line(vis, (w//2, 0), (w//2, h), (50,255,50), 1, cv2.LINE_AA)
        cv2.line(vis, (0, h//2), (w, h//2), (50,255,50), 1, cv2.LINE_AA)

    # 5) small HUD with quick stats
    method = debug_info.get("method") or debug_info.get("alignment", {}).get("method")
    resid  = debug_info.get("residual_px", debug_info.get("alignment", {}).get("residual_px"))
    matched = debug_info.get("n_matched")
    total_tpl = debug_info.get("n_template")
    summary = f"method={method} residual={resid} matched={matched}/{total_tpl} fps={fps:.2f} frame={frame_index}/{total}"

    put_text(vis, "BeeMonitor — NestID Debug", (10, 24), scale=0.7, color=(240,240,240), thick=2)
    put_text(vis, summary, (10, 46), scale=0.55, color=(210,210,210), thick=1)

    ensure_dir(os.path.dirname(out_png) or ".")
    ok = cv2.imwrite(out_png, vis)
    if not ok:
        raise RuntimeError(f"Failed to write {out_png}")

    return {
        "frame": frame_index,
        "fps": fps,
        "total_frames": total,
        "n_tubes": len(tubes),
        "method": method,
        "residual_px": resid,
        "matched": matched,
        "n_template": total_tpl,
        "png": out_png,
    }

# ---------- CLI-ish usage ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("BeeMonitor NestID visual debug")
    ap.add_argument("--cfg", default="beemonitor/config/pipeline.default.yaml", help="pipeline config")
    ap.add_argument("--video", required=True, help="video path")
    ap.add_argument("--frame", type=int, default=0, help="frame index to visualize")
    ap.add_argument("--out", default="outputs/nest_debug/debug.png", help="output PNG path")
    ap.add_argument("--no-centers", action="store_true", help="do not draw template centers")
    ap.add_argument("--no-guides", action="store_true", help="do not draw midline guides")
    args = ap.parse_args()

    info = debug_nest_id(
        cfg_path=args.cfg,
        video_path=args.video,
        out_png=args.out,
        frame_index=args.frame,
        show_template_centers=not args.no_centers,
        draw_guides=not args.no_guides,
    )
    print(json.dumps(info, indent=2))
