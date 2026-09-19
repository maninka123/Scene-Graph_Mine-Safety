from __future__ import annotations

from itertools import permutations
from math import dist
from typing import Iterable

from mine_safety.schemas import SceneEdge, SceneGraph, SceneNode


def build_scene_graph(
    nodes: Iterable[SceneNode | dict],
    graph_id: str = "scene-001",
    edge_distance_m: float = 8.0,
    mean_intensity: float | None = None,
    metadata: dict | None = None,
) -> SceneGraph:
    """Create the paper's directed proximity graph from object/anomaly instances."""
    parsed = [node if isinstance(node, SceneNode) else SceneNode.model_validate(node) for node in nodes]
    edges: list[SceneEdge] = []
    for source, target in permutations(parsed, 2):
        distance_m = dist(source.centroid, target.centroid)
        if distance_m <= edge_distance_m:
            edges.append(
                SceneEdge(
                    source=source.id,
                    target=target.id,
                    distance_m=round(distance_m, 6),
                    relation="near" if distance_m > 1.5 else "very_near",
                )
            )
    return SceneGraph(
        graph_id=graph_id,
        nodes=parsed,
        edges=edges,
        mean_intensity=mean_intensity,
        metadata=metadata or {},
    )


def graph_to_dict(graph: SceneGraph) -> dict:
    return graph.model_dump(mode="json")
