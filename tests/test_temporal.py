from datetime import datetime, timedelta, timezone

from mine_safety.graph.scene import build_scene_graph
from mine_safety.graph.temporal import TemporalGraphTracker


def test_hungarian_tracker_preserves_same_class_identity():
    tracker = TemporalGraphTracker()
    start = datetime.now(timezone.utc)
    first = build_scene_graph([{"id": "frame1-person", "label": "personnel", "centroid": (0, 0, 0)}])
    first.timestamp = start
    tracked_first = tracker.update(first)
    second = build_scene_graph([{"id": "frame2-person", "label": "personnel", "centroid": (.3, 0, 0)}])
    second.timestamp = start + timedelta(seconds=1)
    tracked_second = tracker.update(second)
    assert tracked_first.nodes[0].track_id == tracked_second.nodes[0].track_id
    assert tracked_second.nodes[0].movement_state == "slow"
    assert tracked_second.nodes[0].velocity == (.3, 0.0, 0.0)
