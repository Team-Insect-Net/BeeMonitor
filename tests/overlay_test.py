#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import yaml
from typing import Any, Dict, List

from beemonitor.cli.run_video import run_tracker_on_video
from beemonitor.nestid.factory import build_nest_identifier
from beemonitor.events.factory import build_event_extractor
from beemonitor.analyzer.factory import build_analyzer
from beemonitor.analyzer.exporter import write_visits_csv, write_summary_csv
from beemonitor.overlay import render_overlay_video, Tube


# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _maybe_num(x):
    try:
        return float(x)
    except Exception:
        return None

def tube_from_any(t) -> Tube:
    """
    Accepts a dict or object and returns Tube(tube_id, (x1,y1,x2,y2)).
    Handles many shapes:
      - x1,y1,x2,y2
      - xmin,ymin,xmax,ymax
      - left,top,right,bottom
      - bbox: tuple/list [x1,y1,x2,y2] OR dict {x,y,w,h} / {x1,...}
      - x,y,w,h  (top-left + size)
      - x,y,width,height
      - cx,cy,w,h   (center + size)
      - center:(x,y), size:(w,h)
      - points/poly: [(x,y), ...]  → tight box
    Also tries object attributes with these names.
    """
    # unify to a dict-like view
    if isinstance(t, dict):
        d = t
        get = d.get
        has = lambda k: k in d
    else:
        # object → map attrs
        d = {k: getattr(t, k) for k in dir(t) if not k.startswith("_")}
        def get(k, default=None):
            return d[k] if k in d else default
        def has(k): return k in d

    # tube id candidates
    tid = get("id") or get("tube_id") or get("tid") or get("label") or get("name") or get("idx") or get("grid_id")
    tid = str(tid) if tid is not None else ""

    # 1) direct x1,y1,x2,y2 variants
    for keys in (("x1","y1","x2","y2"), ("xmin","ymin","xmax","ymax"), ("left","top","right","bottom")):
        if all(has(k) for k in keys):
            x1,y1,x2,y2 = map(_maybe_num, (get(keys[0]), get(keys[1]), get(keys[2]), get(keys[3])))
            if None not in (x1,y1,x2,y2):
                return Tube(tube_id=tid, bbox=(x1,y1,x2,y2))

    # 2) bbox field
    if has("bbox"):
        bb = get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            x1,y1,x2,y2 = map(_maybe_num, bb)
            if None not in (x1,y1,x2,y2):
                return Tube(tube_id=tid, bbox=(x1,y1,x2,y2))
        elif isinstance(bb, dict):
            # dict bbox → try x1..; else x,y,w,h
            if all(k in bb for k in ("x1","y1","x2","y2")):
                x1,y1,x2,y2 = map(_maybe_num, (bb["x1"],bb["y1"],bb["x2"],bb["y2"]))
                if None not in (x1,y1,x2,y2):
                    return Tube(tube_id=tid, bbox=(x1,y1,x2,y2))
            if all(k in bb for k in ("x","y","w","h")):
                x,y,w,h = map(_maybe_num, (bb["x"],bb["y"],bb["w"],bb["h"]))
                if None not in (x,y,w,h):
                    return Tube(tube_id=tid, bbox=(x, y, x+w, y+h))

    # 3) x,y,w,h or x,y,width,height
    if all(has(k) for k in ("x","y","w","h")):
        x,y,w,h = map(_maybe_num, (get("x"),get("y"),get("w"),get("h")))
        if None not in (x,y,w,h):
            return Tube(tube_id=tid, bbox=(x, y, x+w, y+h))
    if all(has(k) for k in ("x","y","width","height")):
        x,y,w,h = map(_maybe_num, (get("x"),get("y"),get("width"),get("height")))
        if None not in (x,y,w,h):
            return Tube(tube_id=tid, bbox=(x, y, x+w, y+h))

    # 4) center + size
    if all(has(k) for k in ("cx","cy","w","h")):
        cx,cy,w,h = map(_maybe_num, (get("cx"),get("cy"),get("w"),get("h")))
        if None not in (cx,cy,w,h):
            return Tube(tube_id=tid, bbox=(cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    if has("center") and isinstance(get("center"), (list, tuple)) and all(has(k) for k in ("w","h")):
        cx,cy = map(_maybe_num, get("center"))
        w,h = map(_maybe_num, (get("w"),get("h")))
        if None not in (cx,cy,w,h):
            return Tube(tube_id=tid, bbox=(cx - w/2, cy - h/2, cx + w/2, cy + h/2))

    # 5) polygon/points → tight box
    for key in ("points","poly","polygon","contour"):
        pts = get(key)
        if isinstance(pts, (list, tuple)) and len(pts) >= 3:
            try:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                return Tube(tube_id=tid, bbox=(min(xs), min(ys), max(xs), max(ys)))
            except Exception:
                pass

    # 6) last resort: try to pull known nested fields
    # e.g., get("rect", {}).get("x")...
    for k in ("rect","box","roi"):
        sub = get(k)
        if isinstance(sub, dict):
            if all(s in sub for s in ("x","y","w","h")):
                x,y,w,h = map(_maybe_num, (sub["x"],sub["y"],sub["w"],sub["h"]))
                if None not in (x,y,w,h):
                    return Tube(tube_id=tid, bbox=(x, y, x+w, y+h))
            if all(s in sub for s in ("x1","y1","x2","y2")):
                x1,y1,x2,y2 = map(_maybe_num, (sub["x1"],sub["y1"],sub["x2"],sub["y2"]))
                if None not in (x1,y1,x2,y2):
                    return Tube(tube_id=tid, bbox=(x1,y1,x2,y2))

    # If we reach here, log the unknown shape to help fix upstream
    print("[overlay] Unrecognized tube shape:", type(t), getattr(t, "__dict__", t))
    raise ValueError("Tube missing bbox/x1y1x2y2/x y w h/center+size/points")



# ---------------------------
# Main
# ---------------------------

def main():
    # -- config & inputs --
    cfg_path = "beemonitor/config/pipeline.default.yaml"
    video_path = "videos/test_bee.mp4"   # <-- change as needed
    out_root = "outputs/local_test"

    cfg: Dict[str, Any] = yaml.safe_load(open(cfg_path, "r"))

    # -- 1) Track bees --
    trajectories, fps = run_tracker_on_video(cfg, video_path)
    print(f"FPS: {fps:.2f}")
    print(f"Trajectories: {len(trajectories)}")

    # -- 2) Nest tubes/IDs (stable) --
    nestid = build_nest_identifier(cfg)
    tubes_raw = nestid.detect_or_load(video_path)
    tube_objs: List[Tube] = [tube_from_any(t) for t in tubes_raw]
    print(f"Tubes detected: {len(tube_objs)}")

    # -- 3) IN/OUT events --
    evx = build_event_extractor(cfg)
    events = evx.extract(trajectories, tubes_raw, fps)
    print(f"Events: {len(events)}")

    # -- 4) Visits + per-tube summary --
    analyzer = build_analyzer(cfg)
    visits = analyzer.build_visits(events)
    summaries = analyzer.summarize(visits)
    print(f"Visits: {len(visits)}")

    # -- 5) Write artifacts (JSON/CSV/PNG) --
    run_id = os.path.splitext(os.path.basename(video_path))[0]
    run_dir = os.path.join(out_root, run_id)
    ensure_dir(run_dir)

    write_json(os.path.join(run_dir, "tubes.json"),
               [t.__dict__ if hasattr(t, "__dict__") else t for t in tubes_raw])
    write_json(os.path.join(run_dir, "trajectories.json"), [t.__dict__ for t in trajectories])
    write_json(os.path.join(run_dir, "events.json"), [e.__dict__ for e in events])

    write_visits_csv(os.path.join(run_dir, "visits.csv"), visits)
    write_summary_csv(os.path.join(run_dir, "summary.csv"), summaries)

    # Optional plots (PNG)
    try:
        from beemonitor.analyzer.plot import plot_events, plot_visits
        plot_events(events, title=f"Events — {run_id}", save_path=os.path.join(run_dir, "events.png"))
        plot_visits(visits, title=f"Visits — {run_id}", save_path=os.path.join(run_dir, "visits.png"))
    except Exception as e:
        write_json(os.path.join(run_dir, "plot_error.json"), {"error": str(e)})

    # -- 6) Overlay video (nests + tracks + events) --
    overlay_dir = os.path.join(run_dir, "overlay")
    ensure_dir(overlay_dir)
    overlay_path = os.path.join(overlay_dir, f"{run_id}_overlay.mp4")

    try:
        final_overlay = render_overlay_video(
            video_path=video_path,
            tubes=tube_objs,
            trajectories=trajectories,
            events=events,
            fps=fps,
            out_path=overlay_path,
            draw_tubes=cfg.get("overlay", {}).get("draw_tubes", True),
            draw_tracks=cfg.get("overlay", {}).get("draw_tracks", True),
            tail_len=int(cfg.get("overlay", {}).get("tail_len", 20)),
            every_nth_frame=int(cfg.get("overlay", {}).get("every_nth_frame", 1)),
        )
        print("Overlay written:", final_overlay)
    except Exception as e:
        errp = os.path.join(run_dir, "overlay_error.json")
        write_json(errp, {"error": str(e)})
        print("Overlay failed. See:", errp)

    # -- 7) Meta --
    meta = {
        "run_id": run_id,
        "video": video_path,
        "fps": fps,
        "n_trajectories": len(trajectories),
        "n_events": len(events),
        "n_visits": len(visits),
        "time": __import__("time").time(),
    }
    write_json(os.path.join(run_dir, "run_meta.json"), meta)
    print("Done. Artifacts in:", run_dir)


if __name__ == "__main__":
    main()
