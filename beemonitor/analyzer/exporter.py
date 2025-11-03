from __future__ import annotations
import csv
import os
from typing import Iterable, Any

from beemonitor.analyzer.model import Visit, TubeSummary


def _ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_visits_csv(path: str, visits: Iterable[Visit]) -> None:
    """
    Write per-visit rows.

    New schema columns:
      tube_id, track_id, class_id, species_name, in_frame, out_frame, in_time_s, out_time_s, dwell_s
    """
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tube_id", "track_id", "class_id", "species_name",
            "in_frame", "out_frame", "in_time_s", "out_time_s", "dwell_s"
        ])
        for v in visits:
            w.writerow([
                getattr(v, "tube_id", ""),
                getattr(v, "track_id", ""),
                getattr(v, "class_id", None),
                getattr(v, "species_name", None),
                getattr(v, "in_frame", None),
                getattr(v, "out_frame", None),
                getattr(v, "in_time_s", None),
                getattr(v, "out_time_s", None),
                getattr(v, "dwell_s", None),
            ])


def write_summary_csv(path: str, summaries: Iterable[TubeSummary]) -> None:
    """
    Write per-tube summaries.

    New schema columns (analyzer v2):
      tube_id, class_id, species_name, n_visits, total_dwell_s, mean_dwell_s

    If legacy fields (n_in/n_out/n_complete_visits) are present, a legacy header is written instead.
    """
    _ensure_dir(path)
    summaries = list(summaries)  # allow multiple passes

    # Detect legacy vs new schema by checking attributes on first item
    legacy = False
    if summaries:
        s0 = summaries[0]
        legacy = all(hasattr(s0, a) for a in ("n_in", "n_out", "n_complete_visits"))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        if legacy:
            # Backward-compatible path for old TubeSummary
            w.writerow(["tube_id", "n_in", "n_out", "n_complete_visits",
                        "total_dwell_s", "mean_dwell_s"])
            for s in summaries:
                w.writerow([
                    getattr(s, "tube_id", ""),
                    getattr(s, "n_in", 0),
                    getattr(s, "n_out", 0),
                    getattr(s, "n_complete_visits", 0),
                    getattr(s, "total_dwell_s", 0.0),
                    getattr(s, "mean_dwell_s", 0.0),
                ])
        else:
            # New schema (species-aware)
            w.writerow(["tube_id", "class_id", "species_name",
                        "n_visits", "total_dwell_s", "mean_dwell_s"])
            for s in summaries:
                w.writerow([
                    getattr(s, "tube_id", ""),
                    getattr(s, "class_id", None),
                    getattr(s, "species_name", None),
                    getattr(s, "n_visits", 0),
                    getattr(s, "total_dwell_s", 0.0),
                    getattr(s, "mean_dwell_s", 0.0),
                ])
