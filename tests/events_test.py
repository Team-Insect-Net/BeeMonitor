from beemonitor.events.factory import build_event_extractor
from beemonitor.events.model import Direction
from beemonitor.types import Tube, Trajectory

def _tube(x1,y1,x2,y2, tid="1"):
    return Tube(tube_id=tid, poly=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

def test_simple_entry_exit():
    tubes = [_tube(90,90,110,110,"1")]
    frames = list(range(10))
    bboxes = []
    for f in frames:
        cx = 70 + f*6   # crosses through the tube
        cy = 100
        w, h = 6, 6
        bboxes.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2))
    traj = Trajectory(track_id=7, frames=frames, bboxes=bboxes, confs=[0.9]*len(frames))

    evx = build_event_extractor({"events":{"gate":{
        "in_inflate_px":4, "out_inflate_px":12, "min_dwell_frames":2, "min_gap_frames":3, "flush_on_end": True
    }}})
    events = evx.extract([traj], tubes, fps=30.0)

    assert len(events) == 2
    assert events[0].tube_id == "1" and events[0].direction == Direction.IN
    assert events[1].tube_id == "1" and events[1].direction == Direction.OUT
