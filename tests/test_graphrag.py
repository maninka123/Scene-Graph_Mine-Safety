from mine_safety.graph.scene import build_scene_graph
from mine_safety.reasoning.graphrag import InMemoryGraphArchive, graph_metadata, should_retrieve


def test_archive_requires_linked_identity_or_compatible_anchor():
    archive = InMemoryGraphArchive()
    historical = build_scene_graph([{"id": "worker-01", "label": "personnel", "centroid": (0, 0, 0)}])
    archive.add(historical, [])

    linked = build_scene_graph([{"id": "worker-01", "label": "personnel", "centroid": (1, 0, 0)}])
    assert len(archive.search("personnel", metadata=graph_metadata(linked))) == 1

    unlinked = build_scene_graph([{"id": "worker-99", "label": "personnel", "centroid": (1, 0, 0)}])
    assert archive.search("personnel", metadata=graph_metadata(unlinked)) == []


def test_paper_contextual_pattern_triggers_retrieval():
    assert should_retrieve(contextual_patterns=["developing_over_window"])
    assert not should_retrieve(contextual_patterns=["current_frame"])
