from __future__ import annotations

from pathlib import Path

import numpy as np


def read_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz/rgb from .npz or any Open3D-supported point-cloud file."""
    source = Path(path)
    if source.suffix.lower() == ".npz":
        data = np.load(source)
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
    else:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Install the pointcloud extra to read PCD/PLY files") from exc
        cloud = o3d.io.read_point_cloud(str(source))
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an [N,3] array")
    if colors.shape != points.shape:
        colors = np.zeros_like(points)
    if colors.size and colors.max() > 1:
        colors = colors / 255.0
    return points, np.clip(colors, 0.0, 1.0)


def voxelize(points: np.ndarray, colors: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.floor(points / voxel_size_m).astype(np.int32)
    unique, first, inverse = np.unique(coordinates, axis=0, return_index=True, return_inverse=True)
    # The paper's sparse-tensor contract is XYZRGB: metric coordinates first,
    # followed by RGB values normalised to [0, 1].  Training and inference both
    # call this function, so the feature order stays identical end to end.
    features = np.concatenate([points[first], colors[first]], axis=1).astype(np.float32)
    return unique, features, inverse
