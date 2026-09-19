from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt

from mine_safety.schemas import ReasoningResult, SafetyAlert, SceneGraph


def graph_metadata(graph: SceneGraph, alerts: list[SafetyAlert] | None = None) -> dict:
    return {
        "classes": sorted({node.label for node in graph.nodes}),
        "instance_ids": sorted({node.id for node in graph.nodes}),
        "track_ids": sorted({node.track_id for node in graph.nodes if node.track_id}),
        "spatial_anchor": graph.metadata.get("spatial_anchor"),
        "relations": sorted({edge.relation for edge in graph.edges}),
        "movement_states": sorted({node.movement_state for node in graph.nodes}),
        "safety_flags": sorted({alert.rule for alert in alerts or []}),
        "has_anomaly": any(node.is_anomaly for node in graph.nodes),
    }


def _linked(metadata: dict, candidate: dict) -> bool:
    same_identity = bool(
        set(metadata.get("instance_ids", [])) & set(candidate.get("instance_ids", []))
        or set(metadata.get("track_ids", [])) & set(candidate.get("track_ids", []))
    )
    same_anchor = bool(
        metadata.get("spatial_anchor")
        and metadata.get("spatial_anchor") == candidate.get("spatial_anchor")
    )
    compatible_classes = bool(set(metadata.get("classes", [])) & set(candidate.get("classes", [])))
    compatible_mechanism = bool(
        set(metadata.get("relations", [])) & set(candidate.get("relations", []))
        or set(metadata.get("movement_states", [])) & set(candidate.get("movement_states", []))
        or (metadata.get("has_anomaly") and candidate.get("has_anomaly"))
    )
    return same_identity or (same_anchor and compatible_classes and compatible_mechanism)


def summarise_graph(graph: SceneGraph, alerts: list[SafetyAlert], result: ReasoningResult | None = None) -> str:
    labels: dict[str, int] = {}
    for node in graph.nodes:
        labels[node.label] = labels.get(node.label, 0) + 1
    moving = [node.id for node in graph.nodes if node.movement_state in {"slow", "fast"}]
    anomalies = [node.id for node in graph.nodes if node.is_anomaly]
    risk_interpretations = []
    if result:
        risk_interpretations = [condition.model_dump(mode="json") for condition in result.risk_conditions]
    return json.dumps({
        "graph_id": graph.graph_id,
        "objects": labels,
        "moving": moving,
        "anomalies": anomalies,
        "rules": [alert.rule for alert in alerts],
        "tracks": {node.id: node.track_id for node in graph.nodes if node.track_id},
        "relations": sorted({edge.relation for edge in graph.edges}),
        "spatial_anchor": graph.metadata.get("spatial_anchor"),
        "advisory_risk_interpretations": risk_interpretations,
    }, sort_keys=True)


def should_retrieve(
    *, anomaly_persistence_seconds: float = 0.0, contextual_patterns: list[str] | None = None,
    maximum_similarity: float = 0.0, persistence_trigger: float = 3.0, similarity_trigger: float = 0.70,
) -> bool:
    return (
        anomaly_persistence_seconds > persistence_trigger
        or "developing_over_window" in set(contextual_patterns or [])
        or maximum_similarity > similarity_trigger
    )


def _token_vector(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token in text.lower().replace("_", " ").split():
        vector[token] = vector.get(token, 0.0) + 1.0
    norm = sqrt(sum(value * value for value in vector.values())) or 1.0
    return {key: value / norm for key, value in vector.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(value * b.get(key, 0.0) for key, value in a.items())


@dataclass
class InMemoryGraphArchive:
    """Dependency-light archive for demos/tests; production adapter can use local Qdrant."""

    embed: Callable[[str], object] = _token_vector
    records: list[dict] = field(default_factory=list)

    def add(self, graph: SceneGraph, alerts: list[SafetyAlert], result: ReasoningResult | None = None) -> dict:
        summary = summarise_graph(graph, alerts, result)
        memory = {
            "memory_id": f"memory-{len(self.records) + 1:06d}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph_id": graph.graph_id,
            "summary": summary,
            "metadata": graph_metadata(graph, alerts),
            "embedding": self.embed(summary),
        }
        self.records.append(memory)
        return {key: value for key, value in memory.items() if key != "embedding"}

    def search(self, summary: str, top_k: int = 5, metadata: dict | None = None) -> list[dict]:
        query = self.embed(summary)
        records = self.records
        if metadata is not None:
            records = [item for item in records if _linked(metadata, item["metadata"])]
        ranked = sorted(
            ((float(_cosine(query, item["embedding"])), item) for item in records),
            key=lambda pair: pair[0], reverse=True,
        )
        return [
            {**{key: value for key, value in item.items() if key != "embedding"}, "similarity": score}
            for score, item in ranked[:top_k]
        ]


class QdrantGraphArchive:
    """Production-ready local Qdrant adapter using Qwen3 embedding/reranking models."""

    def __init__(
        self,
        path: str,
        collection: str = "mine_graph_memory",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
    ) -> None:
        from qdrant_client import QdrantClient, models
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.client = QdrantClient(path=path)
        self.collection = collection
        self.models = models
        self.encoder = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.reranker = CrossEncoder(reranker_model, trust_remote_code=True)
        dimension = int(self.encoder.get_sentence_embedding_dimension())
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    def add(self, graph: SceneGraph, alerts: list[SafetyAlert], result: ReasoningResult | None = None) -> dict:
        summary = summarise_graph(graph, alerts, result)
        point_id = str(uuid.uuid4())
        memory_id = f"memory-{point_id}"
        metadata = graph_metadata(graph, alerts)
        payload = {
            "memory_id": memory_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "graph_id": graph.graph_id,
            "summary": summary,
            **metadata,
        }
        vector = self.encoder.encode(summary, normalize_embeddings=True).tolist()
        self.client.upsert(
            collection_name=self.collection,
            points=[self.models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return payload

    def search(self, summary: str, top_k: int = 5, metadata: dict | None = None) -> list[dict]:
        vector = self.encoder.encode(summary, normalize_embeddings=True).tolist()
        candidates = self.client.query_points(
            collection_name=self.collection, query=vector, limit=max(top_k * 8, top_k), with_payload=True,
        ).points
        if metadata is not None:
            candidates = [item for item in candidates if _linked(metadata, dict(item.payload))]
        if not candidates:
            return []
        pairs = [(summary, str(item.payload.get("summary", ""))) for item in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)[:top_k]
        return [
            {**dict(item.payload), "similarity": float(item.score), "reranker_score": float(score)}
            for score, item in ranked
        ]
