from beemonitor.tracking.beetrack import BeeTrack
from beemonitor.types import Detection

trk = BeeTrack(max_age=5, min_hits=2, dist_threshold=40)

# frame 0: one bee
trk.update(0, [Detection(1, bbox=(10,10,20,20), conf=0.9, cls=1)])
# frame 1: near previous
trk.update(1, [Detection(2, bbox=(14,12,24,22), conf=0.9,cls=1)])

trk.update(1, [Detection(2, bbox=(14,12,24,22), conf=0.9,cls=2)])
# frame 2: disappears (no dets)
trk.update(2, [])
# finalize
traj = trk.finalize()
assert len(traj) == 1 and traj[0].track_id == 1
print("OK", traj[0].frames)
