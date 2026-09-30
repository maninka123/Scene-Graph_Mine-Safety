from __future__ import annotations

from itertools import combinations
from math import acos, degrees

import numpy as np
from scipy.ndimage import binary_closing, generate_binary_structure, label
from sklearn.cluster import DBSCAN

from mine_safety.schemas import SceneNode


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    probs = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0])
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    return axis / max(np.linalg.norm(axis), 1e-12)


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(a, b))) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return degrees(acos(np.clip(cosine, -1.0, 1.0)))


def _boundary_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Small anomaly clusters make this exact calculation acceptable and auditable.
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


def _merge_clusters(
    clusters: list[np.ndarray], boundary_m: float, centroid_m: float, orientation_deg: float
) -> list[np.ndarray]:
    changed = True
    while changed:
        changed = False
        for i, j in combinations(range(len(clusters)), 2):
            a, b = clusters[i], clusters[j]
            if (
                np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) <= centroid_m
                and _boundary_distance(a, b) <= boundary_m
                and _angle_degrees(_principal_axis(a), _principal_axis(b)) <= orientation_deg
            ):
                clusters[i] = np.vstack([a, b])
                clusters.pop(j)
                changed = True
                break
        if changed:
            continue
    return clusters


def _remove_small_components(coords: np.ndarray, minimum_voxels: int) -> np.ndarray:
    """Keep 26-connected voxel components that meet the size threshold."""
    remaining = set(map(tuple, coords.tolist()))
    kept: list[tuple[int, int, int]] = []
    neighbours = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while remaining:
        seed = remaining.pop()
        component = [seed]
        for point in component:
            for offset in neighbours:
                adjacent = tuple(point[axis] + offset[axis] for axis in range(3))
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    component.append(adjacent)
        if len(component) >= minimum_voxels:
            kept.extend(component)
    return np.asarray(kept, dtype=int).reshape(-1, 3)


def anomaly_nodes_from_probabilities(
    points: np.ndarray,
    probabilities: np.ndarray,
    *,
    entropy_threshold: float = 0.35,
    minimum_voxels: int = 20,
    epsilon_m: float = 0.10,
    merge_boundary_m: float = 0.15,
    merge_centroid_m: float = 0.20,
    merge_orientation_degrees: float = 15.0,
    voxel_size_m: float = 0.01,
    closing_iterations: int = 1,
) -> list[SceneNode]:
    """Threshold entropy, close, filter connected components, cluster, and merge."""
    xyz = np.asarray(points, dtype=float)
    entropy = predictive_entropy(probabilities)
    high = xyz[entropy > entropy_threshold]
    original_high = high.copy()
    original_high_entropy = entropy[entropy > entropy_threshold]
    if len(high) < minimum_voxels:
        return []
    coords = np.unique(np.floor(high / voxel_size_m).astype(int), axis=0)
    if closing_iterations > 0:
        lower, upper = coords.min(axis=0), coords.max(axis=0)
        shape = tuple((upper - lower + 3).tolist())
        # Bound allocation for unusual, widely separated uncertainty points.
        if int(np.prod(shape)) <= 10_000_000:
            occupancy = np.zeros(shape, dtype=bool)
            local = coords - lower + 1
            occupancy[tuple(local.T)] = True
            closed = binary_closing(occupancy, iterations=closing_iterations)
            component_labels, count = label(closed, structure=generate_binary_structure(3, 3))
            sizes = np.bincount(component_labels.ravel())
            valid = np.zeros(count + 1, dtype=bool)
            valid[1:] = sizes[1:] >= minimum_voxels
            coords = np.argwhere(valid[component_labels]) + lower - 1
        else:
            coords = _remove_small_components(coords, minimum_voxels)
    else:
        coords = _remove_small_components(coords, minimum_voxels)
    if not len(coords):
        return []
    high = (coords.astype(float) + 0.5) * voxel_size_m
    labels = DBSCAN(eps=epsilon_m, min_samples=1).fit_predict(high)
    clusters = [high[labels == label] for label in sorted(set(labels)) if label >= 0]
    clusters = _merge_clusters(clusters, merge_boundary_m, merge_centroid_m, merge_orientation_degrees)
    nodes: list[SceneNode] = []
    for index, cluster in enumerate(clusters, 1):
        mins, maxs = cluster.min(axis=0), cluster.max(axis=0)
        within_cluster_bounds = np.all(
            (original_high >= mins - voxel_size_m) & (original_high <= maxs + voxel_size_m), axis=1
        )
        cluster_entropy = original_high_entropy[within_cluster_bounds]
        if not len(cluster_entropy):
            nearest = int(np.argmin(np.linalg.norm(original_high - cluster.mean(axis=0), axis=1)))
            cluster_entropy = original_high_entropy[[nearest]]
        nodes.append(
            SceneNode(
                id=f"anomaly-{index:03d}",
                label="anomaly",
                centroid=tuple(cluster.mean(axis=0)),
                bbox_dimensions=tuple(maxs - mins),
                orientation=tuple(_principal_axis(cluster)),
                volume_m3=float(np.prod(np.maximum(maxs - mins, 0.0))),
                voxel_count=len(cluster),
                confidence=0.0,
                entropy=float(cluster_entropy.mean()),
                is_anomaly=True,
            )
        )
    return nodes


def close_voxel_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_closing(np.asarray(mask, dtype=bool), iterations=iterations)
