from mine_safety.graph.scene import build_scene_graph


def test_edges_are_directed_and_distance_gated():
    graph = build_scene_graph([
        {"id": "a", "label": "personnel", "centroid": (0, 0, 0)},
        {"id": "b", "label": "mining_vehicle", "centroid": (3, 0, 0)},
        {"id": "c", "label": "floor", "centroid": (20, 0, 0)},
    ], edge_distance_m=8.0)
    assert {(edge.source, edge.target) for edge in graph.edges} == {("a", "b"), ("b", "a")}
    assert all(edge.distance_m == 3.0 for edge in graph.edges)
