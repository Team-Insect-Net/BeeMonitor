import yaml
from beemonitor.cli.run_video import run_tracker_on_video

cfg = yaml.safe_load(open("beemonitor/config/pipeline.default.yaml"))
video_path = "videos/test_bee.mp4"

trajectories, fps = run_tracker_on_video(cfg, video_path)
print("FPS:", fps)
print("Trajectories:", len(trajectories))
