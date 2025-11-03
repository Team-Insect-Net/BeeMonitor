from beemonitor.events.model import Event, Direction
from beemonitor.analyzer.aggregator import EventAnalyzer

def test_visit_pairing_and_dwell():
    # Tube 1: IN at t=10s (track 1), OUT at t=70s (track 2)
    # Tube 2: IN only (still inside)
    evs = [
        Event(tube_id="1", track_id=1, direction=Direction.IN,  frame_in=300, time_in_s=10.0),
        Event(tube_id="1", track_id=2, direction=Direction.OUT, frame_in=2100, time_in_s=70.0),
        Event(tube_id="2", track_id=5, direction=Direction.IN,  frame_in=1500, time_in_s=50.0),
    ]
    an = EventAnalyzer()
    visits = an.build_visits(evs)
    v1 = [v for v in visits if v.tube_id=="1"][0]
    v2 = [v for v in visits if v.tube_id=="2"][0]
    assert v1.dwell_s == 60.0 and v1.out_time_s == 70.0
    assert v2.dwell_s is None and v2.out_time_s is None
    summ = an.summarize(visits)
    s1 = [s for s in summ if s.tube_id=="1"][0]
    s2 = [s for s in summ if s.tube_id=="2"][0]
    assert s1.n_complete_visits == 1 and s1.open_visits == 0
    assert s2.n_complete_visits == 0 and s2.open_visits == 1
