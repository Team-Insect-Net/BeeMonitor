from __future__ import annotations
import os, argparse, csv
from typing import List, Optional
from beemonitor.events.model import Event, Direction
from beemonitor.analyzer.model import Visit
from beemonitor.analyzer.plot import plot_events, plot_visits

def read_events_csv(path: str) -> List[Event]:
    # If you saved JSON for events, you can adapt this; here’s a CSV version if needed later.
    raise NotImplementedError("Use events.json in the run folder; this CLI expects JSON events via run_video output.")

def read_events_json(path: str) -> List[Event]:
    import json
    with open(path, "r") as f:
        raw = json.load(f)
    out = []
    for e in raw:
        out.append(Event(
            tube_id=e["tube_id"],
            track_id=e["track_id"],
            direction=Direction(e["direction"]),
            frame_in=e["frame_in"],
            time_in_s=e["time_in_s"],
            confidence=e.get("confidence", None),
        ))
    return out

def read_visits_csv(path: str) -> List[Visit]:
    out: List[Visit] = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(Visit(
                tube_id=row["tube_id"],
                in_frame=int(row["in_frame"]),
                out_frame=(int(row["out_frame"]) if row["out_frame"] else None),
                in_time_s=float(row["in_time_s"]),
                out_time_s=(float(row["out_time_s"]) if row["out_time_s"] else None),
                dwell_s=(float(row["dwell_s"]) if row["dwell_s"] else None),
                in_track_id=int(row["in_track_id"]),
                out_track_id=(int(row["out_track_id"]) if row["out_track_id"] else None),
            ))
    return out

def main():
    ap = argparse.ArgumentParser(description="BeeMonitor CLI — visualize results from a run folder")
    ap.add_argument("--run-dir", required=True, help="Path to a run folder (contains events.json, visits.csv)")
    ap.add_argument("--title", default=None, help="Plot title override")
    ap.add_argument("--events", action="store_true", help="Plot events PNG")
    ap.add_argument("--visits", action="store_true", help="Plot visits PNG")
    ap.add_argument("--show", action="store_true", help="Show plots instead of saving (useful for notebooks)")
    args = ap.parse_args()

    title = args.title or f"BeeMonitor — {os.path.basename(args.run_dir)}"

    if args.events:
        events_path = os.path.join(args.run_dir, "events.json")
        events = read_events_json(events_path)
        if args.show:
            plot_events(events, title=title, save_path=None)
        else:
            plot_events(events, title=title, save_path=os.path.join(args.run_dir, "events.png"))

    if args.visits:
        visits_path = os.path.join(args.run_dir, "visits.csv")
        visits = read_visits_csv(visits_path)
        if args.show:
            plot_visits(visits, title=title, save_path=None)
        else:
            plot_visits(visits, title=title, save_path=os.path.join(args.run_dir, "visits.png"))

if __name__ == "__main__":
    main()
