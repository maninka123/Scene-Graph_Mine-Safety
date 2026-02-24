from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .augment import augment_points


def _roi_mask(points: np.ndarray, roi_cfg: Dict) -> np.ndarray:
    x_min, x_max = roi_cfg["x"]
    y_min, y_max = roi_cfg["y"]
    z_min, z_max = roi_cfg["z"]
    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )


def _limit_points(
    points: np.ndarray,
    colors: np.ndarray,
    labels: Optional[np.ndarray],
    point_indices: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors, labels, point_indices

    selected = np.random.choice(points.shape[0], size=max_points, replace=False)
    selected.sort()
    points = points[selected]
    colors = colors[selected]
    if labels is not None:
        labels = labels[selected]
    point_indices = point_indices[selected]
    return points, colors, labels, point_indices


def _build_features(points: np.ndarray, colors: np.ndarray, data_cfg: Dict) -> np.ndarray:
    if not bool(data_cfg.get("use_rgb", True)):
        colors = np.zeros_like(colors, dtype=np.float32)
    if data_cfg.get("normalize_rgb", True):
        colors = np.clip(colors, 0.0, 1.0)

    xyz = points.copy()
    if data_cfg.get("normalize_xyz", True):
        roi = data_cfg["roi"]
        scale = np.array(
            [
                max(abs(float(roi["x"][0])), abs(float(roi["x"][1]))),
                max(abs(float(roi["y"][0])), abs(float(roi["y"][1]))),
                max(abs(float(roi["z"][0])), abs(float(roi["z"][1]))),
            ],
            dtype=np.float32,
        )
        scale[scale == 0.0] = 1.0
        xyz = xyz / scale[None, :]

    return np.concatenate([xyz, colors], axis=1).astype(np.float32)


def voxelize(
    points: np.ndarray,
    features: np.ndarray,
    labels: Optional[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    coords = np.floor(points / voxel_size).astype(np.int32)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    voxel_count = unique_coords.shape[0]

    feat_sum = np.zeros((voxel_count, features.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    for idx in range(features.shape[1]):
        np.add.at(feat_sum[:, idx], inverse, features[:, idx])
    voxel_features = (feat_sum / counts[:, None]).astype(np.float32)

    voxel_labels: Optional[np.ndarray] = None
    if labels is not None:
        voxel_labels = np.full(voxel_count, -1, dtype=np.int64)
        for vid in range(voxel_count):
            label_values = labels[inverse == vid]
            label_values = label_values[label_values >= 0]
            if label_values.size > 0:
                voxel_labels[vid] = np.bincount(label_values).argmax()

    return unique_coords, voxel_features, voxel_labels, inverse


def preprocess_sample(
    points: np.ndarray,
    colors: np.ndarray,
    labels: Optional[np.ndarray],
    data_cfg: Dict,
    max_points: int,
    augmentation_cfg: Optional[Dict] = None,
) -> Dict:
    original_count = points.shape[0]
    original_indices = np.arange(original_count, dtype=np.int64)
    original_points = points
    original_colors = colors
    original_labels = labels.copy() if labels is not None else None

    roi_mask = _roi_mask(points, data_cfg["roi"])
    points = points[roi_mask]
    colors = colors[roi_mask]
    point_indices = original_indices[roi_mask]
    if labels is not None:
        labels = labels[roi_mask]

    # Some frames may become empty after ROI filtering. Fall back to full-frame points
    # so dataloader/model do not crash on empty sparse tensors.
    if points.shape[0] == 0:
        points = original_points
        colors = original_colors
        point_indices = original_indices
        labels = original_labels

    points, colors, labels, point_indices = _limit_points(
        points=points,
        colors=colors,
        labels=labels,
        point_indices=point_indices,
        max_points=max_points,
    )

    if augmentation_cfg:
        points, colors, labels = augment_points(points, colors, labels, augmentation_cfg)

    # Guard against rare augmentation/dropout corner-cases.
    if points.shape[0] == 0:
        points = original_points
        colors = original_colors
        point_indices = original_indices
        labels = original_labels

    features = _build_features(points, colors, data_cfg)
    coords, voxel_features, voxel_labels, inverse = voxelize(
        points=points,
        features=features,
        labels=labels,
        voxel_size=float(data_cfg["voxel_size"]),
    )

    return {
        "coords": coords.astype(np.int32),
        "features": voxel_features.astype(np.float32),
        "voxel_labels": voxel_labels,
        "inverse_indices": inverse.astype(np.int64),
        "points": points.astype(np.float32),
        "colors": colors.astype(np.float32),
        "point_indices": point_indices.astype(np.int64),
        "original_count": int(original_count),
    }
