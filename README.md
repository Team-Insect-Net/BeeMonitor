
# BeeMonitor

Organized repository skeleton for the BeeMonitor system (solitary-bee foraging and nesting behavior).  
This layout preserves your current implementation under `beemonitor/legacy/` while preparing modular packages for IO, detection, tracking, nest identification, event logic, and synthesis.

## Quick start (development)
```bash
pip install -e .
python scripts/process_video.py --video path/to/clip.mp4
```

## Layout
- `beemonitor/legacy/` — your current scripts, unchanged
- `beemonitor/config/` — YAML configs (pipeline defaults, tracker)
- `beemonitor/types.py` — shared dataclasses for clean interfaces
- `beemonitor/{io,detect,tracking,nestid,events,synth}/` — new modular packages (to be filled as we refactor)
- `scripts/` — CLI entry points
- `tests/` — unit & regression tests (TBD)

## Next steps
- Standardize tracker outputs to `Trajectory`
- Move nest detection and event logic into `nestid/` and `events/`
- Add tests + small golden dataset for CI
```