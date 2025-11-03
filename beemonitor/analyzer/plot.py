from __future__ import annotations
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from beemonitor.analyzer.model import Visit
from beemonitor.events.model import Event, Direction


def plot_visits(
    visits: List[Visit],
    title: str = "BeeMonitor — Visits Timeline",
    show_events: bool = True,
    figsize=(10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot IN→OUT visit intervals per tube as horizontal bars (one figure).
    - X-axis: time (seconds)
    - Y-axis: tubes (one row per tube)
    - Optional: overlay raw IN/OUT event markers

    Notes:
      * Uses default matplotlib colors (none specified).
      * Exactly one plot per call (no subplots).
    """
    if not visits:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tube")
        ax.text(0.5, 0.5, "No visits", ha="center", va="center", transform=ax.transAxes)
        _finalize(fig, save_path)
        return

    # Group visits by tube and preserve order
    tubes = sorted(list({v.tube_id for v in visits}))
    tube_to_row = {tid: i for i, tid in enumerate(tubes)}

    fig, ax = plt.subplots(figsize=figsize)

    # draw each visit as a horizontal bar (open visits extend with a small cap)
    for v in visits:
        row = tube_to_row[v.tube_id]
        x0 = v.in_time_s
        x1 = v.out_time_s if v.out_time_s is not None else v.in_time_s  # draw as point if open
        # bar height
        y0 = row - 0.35
        height = 0.7

        # Use default color cycle by not specifying color
        if v.out_time_s is None:
            # open visit: draw a short bar/marker at x0
            ax.hlines(y=row, xmin=x0, xmax=x0, linewidth=6)  # a vertical tick-like marker
        else:
            ax.broken_barh([(x0, max(0.0, x1 - x0))], (y0, height))

    # Axis labels & ticks
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(range(len(tubes)))
    ax.set_yticklabels(tubes)
    ax.set_ylim(-1, len(tubes))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

    # Optionally overlay discrete IN/OUT markers from reconstructed events
    if show_events:
        # Reconstruct events from visits (if you have raw events, use plot_events instead)
        for v in visits:
            row = tube_to_row[v.tube_id]
            # IN marker
            ax.plot([v.in_time_s], [row], marker="o", markersize=5)
            # OUT marker
            if v.out_time_s is not None:
                ax.plot([v.out_time_s], [row], marker="x", markersize=6)

    _finalize(fig, save_path)


def plot_events(
    events: List[Event],
    title: str = "BeeMonitor — IN/OUT Events",
    figsize=(10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot raw events (IN circles, OUT crosses) on a time vs tube chart (one figure).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not events:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tube")
        ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
        _finalize(fig, save_path)
        return

    tubes = sorted(list({e.tube_id for e in events}))
    tube_to_row = {tid: i for i, tid in enumerate(tubes)}

    for e in events:
        row = tube_to_row[e.tube_id]
        if e.direction == Direction.IN:
            ax.plot([e.time_in_s], [row], marker="o", markersize=5)
        else:
            ax.plot([e.time_in_s], [row], marker="x", markersize=6)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(range(len(tubes)))
    ax.set_yticklabels(tubes)
    ax.set_ylim(-1, len(tubes))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

    _finalize(fig, save_path)


def _finalize(fig, save_path: Optional[str]):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
