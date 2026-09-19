from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from mine_safety.schemas import SceneNode


def _orientation(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) < 3:
        return (1.0, 0.0, 0.0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, np.argmax(values)]
    axis /= max(np.linalg.norm(axis), 1e-12)
    return tuple(float(value) for value in axis)


def extract_instances(
    points: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    epsilon_m: float = 0.25,
    minimum_voxels: int = 20,
) -> list[SceneNode]:
    """Convert voxel predictions to class-consistent object instances."""
    nodes: list[SceneNode] = []
    entropy = -(np.clip(probabilities, 1e-12, 1) * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1)
    for class_id, class_name in enumerate(class_names):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < minimum_voxels:
            continue
        clustering = DBSCAN(eps=epsilon_m, min_samples=minimum_voxels).fit_predict(points[indices])
        for instance_index, cluster_id in enumerate(sorted(set(clustering) - {-1}), 1):
            member_indices = indices[clustering == cluster_id]
            cluster = points[member_indices]
            minimum, maximum = cluster.min(axis=0), cluster.max(axis=0)
            dimensions = maximum - minimum
            nodes.append(SceneNode(
                id=f"{class_name}-{instance_index:03d}", label=class_name,
                centroid=tuple(cluster.mean(axis=0)), bbox_dimensions=tuple(dimensions),
                orientation=_orientation(cluster), volume_m3=float(np.prod(dimensions)),
                voxel_count=len(cluster), confidence=float(probabilities[member_indices, class_id].mean()),
                entropy=float(entropy[member_indices].mean()),
            ))
    return nodes
