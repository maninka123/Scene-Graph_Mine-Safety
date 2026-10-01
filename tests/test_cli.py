import json
import sys
from pathlib import Path

from mine_safety import cli


def test_qdrant_flag_selects_configured_archive(monkeypatch, tmp_path):
    received = {}

    class FakeArchive:
        def __init__(self, **kwargs):
            received["archive_args"] = kwargs

    class FakePipeline:
        def __init__(self, *, config, archive):
            received["archive"] = archive

        def process_graph(self, graph, *, run_llm):
            received["run_llm"] = run_llm
            return {"graph_id": graph.graph_id}

    monkeypatch.setattr(cli, "QdrantGraphArchive", FakeArchive)
    monkeypatch.setattr(cli, "MineSafetyPipeline", FakePipeline)
    output = tmp_path / "assessment.json"
    memory = tmp_path / "qdrant"
    scene = Path(__file__).resolve().parents[1] / "examples" / "scene_graph.json"
    monkeypatch.setattr(sys, "argv", ["mine-safety", str(scene), "--llm", "--qdrant-path", str(memory), "--output", str(output)])

    cli.main()

    assert isinstance(received["archive"], FakeArchive)
    assert received["archive_args"] == {
        "path": str(memory),
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
    }
    assert received["run_llm"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["graph_id"]
