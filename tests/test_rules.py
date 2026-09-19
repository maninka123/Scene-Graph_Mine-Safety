from datetime import datetime, timedelta, timezone

from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.rules import SafetyRuleEngine


def test_static_proximity_uses_bounding_box_clearance():
    graph = build_scene_graph([
        {"id": "p", "label": "personnel", "centroid": (2.3, 0, 0), "bbox_dimensions": (.5, .5, 1.8)},
        {"id": "m", "label": "mining_vehicle", "centroid": (0, 0, 0), "bbox_dimensions": (3, 1.8, 1.8)},
    ])
    alerts = SafetyRuleEngine(load_config()["rules"]).evaluate(graph)
    assert "proximity_violation" in {alert.rule for alert in alerts}
    flagged = {flag for edge in graph.edges for flag in edge.safety_flags}
    assert "proximity_violation" in flagged


def test_three_people_trigger_density_rule():
    nodes = [{"id": "m", "label": "mining_vehicle", "centroid": (0, 0, 0)}]
    nodes += [{"id": f"p{i}", "label": "personnel", "centroid": (2.5, i - 1, 0)} for i in range(3)]
    alerts = SafetyRuleEngine(load_config()["rules"]).evaluate(build_scene_graph(nodes))
    assert "congestion" in {alert.rule for alert in alerts}


def test_visibility_requires_full_rolling_window():
    engine = SafetyRuleEngine(load_config()["rules"])
    first = build_scene_graph([], mean_intensity=10)
    first.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert "low_visibility" not in {alert.rule for alert in engine.evaluate(first)}

    later = build_scene_graph([], mean_intensity=10)
    later.timestamp = first.timestamp + timedelta(seconds=3)
    assert "low_visibility" in {alert.rule for alert in engine.evaluate(later)}
