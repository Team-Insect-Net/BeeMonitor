# BeeMonitor

BeeMonitor is a hardware and software system for studying solitary bee foraging and nesting behavior.

## What you can do

- Detect bees and nest tubes (YOLO)
- Track bees (BeeTrack)
- Assign consistent nest IDs across videos
- Extract entry/exit events
- Build visits and compute dwell times
- Visualize timelines
- Retrain detectors on your dataset

## Quickstart
```bash
# process one video end-to-end
python -m beemonitor.cli.run_video --video data/clip.mp4 --outdir outputs/runs
