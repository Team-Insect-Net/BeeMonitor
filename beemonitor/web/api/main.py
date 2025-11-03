from __future__ import annotations

import os
import json
import time
import tempfile
from typing import Optional, List, Dict, Any

import yaml
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, Query,
    Depends, Security, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
import uvicorn

# --- BeeMonitor imports (explicit to avoid __init__ re-export issues) ---
from beemonitor.nestid.factory import build_nest_identifier
from beemonitor.events.factory import build_event_extractor
from beemonitor.analyzer.factory import build_analyzer
from beemonitor.analyzer.exporter import write_visits_csv, write_summary_csv
from beemonitor.cli.run_video import run_tracker_on_video
from beemonitor.overlay import render_overlay_video, Tube


# =============================================================================
# Configuration
# =============================================================================

CFG_PATH = "beemonitor/config/pipeline.default.yaml"
OUT_ROOT = "outputs/api_runs"
os.makedirs(OUT_ROOT, exist_ok=True)

RUN_INDEX = os.path.join(OUT_ROOT, "_index.json")

MAX_UPLOAD_MB = int(os.getenv("BEEMONITOR_MAX_UPLOAD_MB", "512"))
CORS_ORIGINS = os.getenv("BEEMONITOR_CORS_ORIGINS", "*").split(",")

API_KEYS: set[str] = {
    k.strip() for k in os.getenv("BEEMONITOR_API_KEYS", "").split(",") if k.strip()
}
STRICT_KEYS = os.getenv("BEEMONITOR_STRICT_KEYS", "true").lower() in {"1", "true", "yes"}
if STRICT_KEYS and not API_KEYS:
    raise RuntimeError(
        "BeeMonitor API: no API keys configured. "
        "Set BEEMONITOR_API_KEYS='key1,key2' (or BEEMONITOR_STRICT_KEYS=false for dev)."
    )


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="BeeMonitor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS] if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=False)


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> bool:
    if not API_KEYS and not STRICT_KEYS:
        return True  # dev mode
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Api-Key"},
        )
    return True


# =============================================================================
# Helpers
# =============================================================================

def _load_cfg() -> Dict[str, Any]:
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def _redact_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    red = json.loads(json.dumps(cfg))  # deep copy
    if "detectors" in red:
        for _, section in red["detectors"].items():
            if isinstance(section, dict) and "weights" in section:
                section["weights"] = "<redacted>"
    return red


def _safe_path(base: str, rel: str) -> str:
    full = os.path.abspath(os.path.join(base, rel))
    base_abs = os.path.abspath(base)
    if full != base_abs and not full.startswith(base_abs + os.sep):
        raise HTTPException(status_code=400, detail="Invalid path")
    return full


def _index_load() -> Dict[str, Any]:
    if not os.path.exists(RUN_INDEX):
        return {"runs": []}
    with open(RUN_INDEX, "r") as f:
        return json.load(f)


def _index_append(entry: Dict[str, Any]) -> None:
    idx = _index_load()
    idx["runs"].append(entry)
    with open(RUN_INDEX, "w") as f:
        json.dump(idx, f, indent=2)


def _run_id_from_video(video_path_or_name: str) -> str:
    base = os.path.splitext(os.path.basename(video_path_or_name))[0]
    return base


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _tube_from_any(t: Any) -> Tube:
    """
    Accepts either:
      - object with .id and .bbox OR .x1/.y1/.x2/.y2
      - dict with keys 'id' and 'bbox' OR x1/y1/x2/y2
    Returns Tube(tube_id, (x1,y1,x2,y2)).
    """
    if isinstance(t, dict):
        tid = str(t.get("id", t.get("tube_id", "")))
        if "bbox" in t and t["bbox"] is not None:
            x1, y1, x2, y2 = t["bbox"]
        else:
            x1, y1, x2, y2 = t.get("x1"), t.get("y1"), t.get("x2"), t.get("y2")
    else:
        tid = str(getattr(t, "id", getattr(t, "tube_id", "")))
        if hasattr(t, "bbox") and getattr(t, "bbox") is not None:
            x1, y1, x2, y2 = t.bbox
        else:
            x1, y1, x2, y2 = getattr(t, "x1", None), getattr(t, "y1", None), getattr(t, "x2", None), getattr(t, "y2", None)

    if None in (x1, y1, x2, y2):
        raise ValueError("Tube missing bbox coordinates")
    return Tube(tube_id=tid, bbox=(float(x1), float(y1), float(x2), float(y2)))


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": time.time()}


@app.get("/config")
def get_config(_: bool = Depends(require_api_key)) -> Dict[str, Any]:
    cfg = _load_cfg()
    return {"config": _redact_cfg(cfg)}


@app.get("/runs")
def list_runs(_: bool = Depends(require_api_key)) -> Dict[str, Any]:
    return _index_load()


@app.get("/runs/{run_id}/files")
def list_run_files(run_id: str, _: bool = Depends(require_api_key)) -> Dict[str, Any]:
    run_dir = _safe_path(OUT_ROOT, run_id)
    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    files: List[str] = []
    for root, _, names in os.walk(run_dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), run_dir)
            files.append(rel)
    return {"run_id": run_id, "files": sorted(files)}


@app.get("/runs/{run_id}/download")
def download_artifact(
    run_id: str,
    path: str = Query(..., description="Relative path under the run directory"),
    _: bool = Depends(require_api_key),
):
    run_dir = _safe_path(OUT_ROOT, run_id)
    fpath = _safe_path(run_dir, path)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fpath)


@app.post("/run")
async def run_uploaded(
    video: UploadFile = File(...),
    make_plots: bool = Form(False),
    _: bool = Depends(require_api_key),
) -> Dict[str, Any]:
    data = await video.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_MB} MB).")

    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = _run_common(tmp_path, make_plots=make_plots, uploaded_name=video.filename)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return result


@app.post("/run-by-path")
def run_by_path(
    video_path: str = Form(...),
    make_plots: bool = Form(False),
    _: bool = Depends(require_api_key),
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return _run_common(video_path, make_plots=make_plots, uploaded_name=None)


# =============================================================================
# Core processing
# =============================================================================

def _run_common(vpath: str, make_plots: bool, uploaded_name: Optional[str]) -> Dict[str, Any]:
    cfg = _load_cfg()

    # 1) Nest IDs (stable tube assignment)
    nestid = build_nest_identifier(cfg)
    tubes = nestid.detect_or_load(vpath)  # list[object|dict] with id + bbox or x1..y2

    # 2) Tracking (bee trajectories)
    trajectories, fps = run_tracker_on_video(cfg, vpath)

    # 3) Events
    evx = build_event_extractor(cfg)
    events = evx.extract(trajectories, tubes, fps)

    # 4) Visits + per-tube summary
    analyzer = build_analyzer(cfg)
    visits = analyzer.build_visits(events)
    summaries = analyzer.summarize(visits)

    # 5) Write artifacts
    run_id = _run_id_from_video(uploaded_name or vpath)
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    _write_json(os.path.join(run_dir, "tubes.json"), [t.__dict__ if hasattr(t, "__dict__") else t for t in tubes])
    _write_json(os.path.join(run_dir, "trajectories.json"), [t.__dict__ for t in trajectories])
    _write_json(os.path.join(run_dir, "events.json"), [e.__dict__ for e in events])

    visits_csv = os.path.join(run_dir, "visits.csv")
    summary_csv = os.path.join(run_dir, "summary.csv")
    write_visits_csv(visits_csv, visits)
    write_summary_csv(summary_csv, summaries)

    # 6) Optional plots
    if make_plots:
        from beemonitor.analyzer.plot import plot_events, plot_visits
        plot_events(events, title=f"Events — {run_id}", save_path=os.path.join(run_dir, "events.png"))
        plot_visits(visits, title=f"Visits — {run_id}", save_path=os.path.join(run_dir, "visits.png"))

    # 7) Overlay video (tubes + tracks + events)
    overlay_path = None
    if (cfg.get("output", {}).get("make_overlay_video", True)):
        try:
            tube_objs: List[Tube] = [_tube_from_any(t) for t in tubes]
            overlay_dir = os.path.join(run_dir, "overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_path = os.path.join(overlay_dir, f"{run_id}_overlay.mp4")
            render_overlay_video(
                video_path=vpath,
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
        except Exception as e:
            # don't fail the run if overlay rendering has a hiccup
            _write_json(os.path.join(run_dir, "overlay_error.json"), {"error": str(e)})

    # 8) Meta + index
    meta = {
        "run_id": run_id,
        "video": uploaded_name or vpath,
        "fps": fps,
        "n_trajectories": len(trajectories),
        "n_events": len(events),
        "n_visits": len(visits),
        "plots": bool(make_plots),
        "overlay_video": overlay_path,
        "time": time.time(),
    }
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)
    _index_append(meta | {"run_dir": run_dir})

    return {"ok": True, **meta}


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "9099")))
