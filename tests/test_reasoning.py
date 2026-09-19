import pytest

from mine_safety.graph.scene import build_scene_graph
from mine_safety.reasoning.llm import LocalQwenReasoner


def test_mock_reasoner_accepts_grounded_json():
    graph = build_scene_graph([{"id": "a", "label": "anomaly", "centroid": (0, 0, 0), "is_anomaly": True}])
    generator = lambda _: '{"hazards":[{"condition":"unexpected obstacle","object_ids":["a"],"severity":"medium","status":"observed","explanation":"high entropy object","current_evidence":["a is anomaly"],"memory_ids":[],"historical_evidence":[]}],"summary":"one contextual condition","no_additional_hazard":false}'
    config = {"contextual_model": "unused", "max_new_tokens": 500, "temperature": .15, "top_p": .9, "regeneration_attempts": 1}
    result = LocalQwenReasoner(config, generator=generator).reason(graph, {}, [])
    assert result.hazards[0].object_ids == ["a"]


def test_mock_reasoner_rejects_hallucinated_object():
    graph = build_scene_graph([{"id": "a", "label": "floor", "centroid": (0, 0, 0)}])
    generator = lambda _: '{"hazards":[{"condition":"x","object_ids":["missing"],"severity":"low","status":"observed","explanation":"x","current_evidence":["x"],"memory_ids":[],"historical_evidence":[]}],"summary":"x","no_additional_hazard":false}'
    config = {"contextual_model": "unused", "max_new_tokens": 500, "temperature": .15, "top_p": .9, "regeneration_attempts": 1}
    with pytest.raises(RuntimeError, match="unknown object"):
        LocalQwenReasoner(config, generator=generator).reason(graph, {}, [])
