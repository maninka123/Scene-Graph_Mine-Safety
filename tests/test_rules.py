from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.rules import SafetyRuleEngine


def test_static_proximity_uses_bounding_box_clearance():
    graph = build_scene_graph([
        {"id": "p", "label": "personnel", "centroid": (2.3, 0, 0), "bbox_dimensions": (.5, .5, 1.8)},
        {"id": "m", "label": "mining_vehicle", "centroid": (0, 0, 0), "bbox_dimensions": (3, 1.8, 1.8)},
    ])
    alerts = SafetyRuleEngine(load_config()["rules"]).evaluate(graph)
    assert "static_proximity" in {alert.rule for alert in alerts}


def test_three_people_trigger_density_rule():
    nodes = [{"id": "m", "label": "mining_vehicle", "centroid": (0, 0, 0)}]
    nodes += [{"id": f"p{i}", "label": "personnel", "centroid": (2.5, i - 1, 0)} for i in range(3)]
    alerts = SafetyRuleEngine(load_config()["rules"]).evaluate(build_scene_graph(nodes))
    assert "operational_congestion" in {alert.rule for alert in alerts}
