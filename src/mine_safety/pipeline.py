from __future__ import annotations

from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.graph.temporal import TemporalGraphTracker
from mine_safety.reasoning.graphrag import InMemoryGraphArchive, should_retrieve, summarise_graph
from mine_safety.reasoning.llm import LocalQwenReasoner
from mine_safety.rules import SafetyRuleEngine
from mine_safety.schemas import ReasoningResult, SceneGraph, SceneNode


class MineSafetyPipeline:
    """Orchestrates graph construction, tracking, rules, local LLM, and memory."""

    def __init__(
        self, config: dict | None = None, reasoner: LocalQwenReasoner | None = None, archive=None
    ) -> None:
        self.config = config or load_config()
        graph_cfg = self.config["graph"]
        self.tracker = TemporalGraphTracker(
            window_seconds=graph_cfg["temporal_window_seconds"],
            label_penalty=graph_cfg["association_label_penalty"],
            max_displacement_m=graph_cfg["association_max_displacement_m"],
            stationary_m=graph_cfg["stationary_displacement_m"],
            slow_m=graph_cfg["slow_displacement_m"],
        )
        self.rules = SafetyRuleEngine(self.config["rules"])
        self.reasoner = reasoner
        self.archive = archive or InMemoryGraphArchive()

    def process_graph(self, graph: SceneGraph, run_llm: bool = False) -> dict:
        tracked = self.tracker.update(graph)
        temporal = self.tracker.snapshot()
        alerts = self.rules.evaluate(tracked)
        contextual: ReasoningResult | None = None
        longitudinal: ReasoningResult | None = None
        retrieved: list[dict] = []
        if run_llm:
            if self.reasoner is None:
                self.reasoner = LocalQwenReasoner(self.config["reasoning"])
            contextual = self.reasoner.reason(tracked, temporal, alerts)
            candidates = self.archive.search(
                summarise_graph(tracked, alerts, contextual), self.config["reasoning"]["retrieval_top_k"]
            )
            maximum_similarity = max((item["similarity"] for item in candidates), default=0.0)
            anomaly_duration = self._maximum_anomaly_duration()
            if should_retrieve(
                anomaly_persistence_seconds=anomaly_duration,
                contextual_statuses=[item.status for item in contextual.hazards],
                maximum_similarity=maximum_similarity,
                persistence_trigger=self.config["reasoning"]["anomaly_persistence_seconds"],
                similarity_trigger=self.config["reasoning"]["similarity_trigger"],
            ):
                retrieved = candidates
                longitudinal = self.reasoner.reason(tracked, temporal, alerts, memories=retrieved)
        memory = self.archive.add(tracked, alerts, longitudinal or contextual)
        return {
            "scene_graph": tracked.model_dump(mode="json"),
            "temporal_graph": temporal,
            "deterministic_alerts": [alert.model_dump(mode="json") for alert in alerts],
            "contextual_reasoning": contextual.model_dump(mode="json") if contextual else None,
            "retrieved_memories": retrieved,
            "longitudinal_reasoning": longitudinal.model_dump(mode="json") if longitudinal else None,
            "memory_unit": memory,
        }

    def process_nodes(
        self, nodes: list[SceneNode | dict], graph_id: str = "scene-001",
        mean_intensity: float | None = None, run_llm: bool = False,
    ) -> dict:
        graph = build_scene_graph(
            nodes, graph_id=graph_id, edge_distance_m=self.config["graph"]["edge_distance_m"],
            mean_intensity=mean_intensity,
        )
        return self.process_graph(graph, run_llm=run_llm)

    def _maximum_anomaly_duration(self) -> float:
        from datetime import datetime

        duration = 0.0
        for observations in self.tracker.snapshot()["tracks"].values():
            anomalous = [item for item in observations if item["is_anomaly"]]
            if len(anomalous) > 1:
                duration = max(duration, (
                    datetime.fromisoformat(anomalous[-1]["timestamp"])
                    - datetime.fromisoformat(anomalous[0]["timestamp"])
                ).total_seconds())
        return duration
