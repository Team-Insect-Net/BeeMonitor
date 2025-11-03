from __future__ import annotations
import os, json, tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import yaml

from beemonitor.analyzer.factory import build_analyzer
from beemonitor.events.factory import build_event_extractor
from beemonitor.nestid.factory import build_nest_identifier
from beemonitor.cli.run_video import run_tracker_on_video  # reuse the loop

app = FastAPI(title="BeeMonitor API")

CFG_PATH = "beemonitor/config/pipeline.default.yaml"
OUT_ROOT = "outputs/api_runs"
os.makedirs(OUT_ROOT, exist_ok=True)

@app.post("/run")
async def run(video: UploadFile = File(...), make_plots: bool = Form(False)):
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
        data = await video.read()
        tmp.write(data)
        tmp_path = tmp.name

    cfg = yaml.safe_load(open(CFG_PATH, "r"))

    # 1) nest ids (cached per video name isn't meaningful for temp file; just do detect_tubes)
    nestid = build_nest_identifier(cfg)
    # For the API: detect on first frame only (no cache path), use internal detect_tubes()
    # We'll reuse detect_or_load by symlinking a name
    video_name = os.path.splitext(os.path.basename(tmp_path))[0]

    # 2) tracker
    trajectories, fps = run_tracker_on_video(cfg, tmp_path)

    # 3) tubes
    tubes = nestid.detect_or_load(tmp_path)

    # 4) events
    evx = build_event_extractor(cfg)
    events = evx.extract(trajectories, tubes, fps)

    # 5) visits
    analyzer = build_analyzer(cfg)
    visits = analyzer.build_visits(events)
    summaries = analyzer.summarize(visits)

    run_dir = os.path.join(OUT_ROOT, video_name)
    os.makedirs(run_dir, exist_ok=True)

    json.dump([e.__dict__ for e in events], open(os.path.join(run_dir, "events.json"), "w"), indent=2)
    import csv
    from beemonitor.analyzer.exporter import write_visits_csv, write_summary_csv
    write_visits_csv(os.path.join(run_dir, "visits.csv"), visits)
    write_summary_csv(os.path.join(run_dir, "summary.csv"), summaries)

    return {
        "run_dir": run_dir,
        "n_events": len(events),
        "n_visits": len(visits),
        "fps": fps
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099)
