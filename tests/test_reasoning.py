import pytest

from mine_safety.graph.scene import build_scene_graph
from mine_safety.reasoning.llm import LocalQwenReasoner


def test_mock_reasoner_accepts_grounded_json():
    graph = build_scene_graph([{"id": "a", "label": "anomaly", "centroid": (0, 0, 0), "is_anomaly": True}])
    generator = lambda _: '{"hazard_detected":true,"risk_conditions":[{"condition":"persistent anomaly cluster","object_ids":["a"],"evidence":"a persists across multiple graph frames with increasing entropy.","severity":"medium","temporal_pattern":"developing_over_window"}],"explanation":"A persistent anomaly pattern is developing within the current observation window."}'
    config = {"contextual_model": "unused", "max_new_tokens": 500, "temperature": .15, "top_p": .9, "regeneration_attempts": 1}
    result = LocalQwenReasoner(config, generator=generator).reason(graph, {}, [])
    assert result.risk_conditions[0].object_ids == ["a"]


def test_mock_reasoner_rejects_hallucinated_object():
    graph = build_scene_graph([{"id": "a", "label": "floor", "centroid": (0, 0, 0)}])
    generator = lambda _: '{"hazard_detected":true,"risk_conditions":[{"condition":"unsupported condition","object_ids":["missing"],"evidence":"A missing object is claimed as evidence.","severity":"low","temporal_pattern":"current_frame"}],"explanation":"An unsupported condition was claimed."}'
    config = {"contextual_model": "unused", "max_new_tokens": 500, "temperature": .15, "top_p": .9, "regeneration_attempts": 1}
    with pytest.raises(RuntimeError, match="unknown object"):
        LocalQwenReasoner(config, generator=generator).reason(graph, {}, [])


def test_longitudinal_schema_requires_grounded_memory():
    graph = build_scene_graph([{"id": "a", "label": "anomaly", "centroid": (0, 0, 0), "is_anomaly": True}])
    generator = lambda _: '{"hazard_detected":true,"risk_conditions":[{"condition":"recurring anomaly pattern","object_ids":["a"],"memory_ids":["missing-memory"],"current_evidence":"a has a persistent anomaly pattern in the current graph.","historical_evidence":"The referenced memory reports the same anomaly mechanism.","severity":"medium","temporal_pattern":"recurring"}],"explanation":"The anomaly pattern recurs across current and historical evidence."}'
    config = {"contextual_model": "unused", "max_new_tokens": 500, "temperature": .15, "top_p": .9, "regeneration_attempts": 0}
    with pytest.raises(RuntimeError, match="unknown memory"):
        LocalQwenReasoner(config, generator=generator).reason(
            graph, {}, [], memories=[{"memory_id": "memory-000001"}]
        )
