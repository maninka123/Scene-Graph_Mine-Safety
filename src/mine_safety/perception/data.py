from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from mine_safety.perception.io import voxelize


def augment_points(
    points: np.ndarray, colors: np.ndarray, *, rotation_degrees: float = 30.0,
    scale: tuple[float, float] = (0.9, 1.1), jitter_std_m: float = 0.01,
    point_dropout_max: float = 0.10, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    angle = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees))
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformed = points @ rotation.T * rng.uniform(*scale)
    transformed += rng.normal(0.0, jitter_std_m, transformed.shape)
    keep = rng.random(len(points)) >= rng.uniform(0.0, point_dropout_max)
    return transformed[keep].astype(np.float32), colors[keep].astype(np.float32)


class NpzPointCloudDataset(Dataset):
    """Manifest-backed dataset. Each NPZ contains points, colors, and optional labels."""

    def __init__(self, manifest: str | Path) -> None:
        base = Path(manifest).resolve().parent
        self.records = []
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                source = Path(record["path"])
                record["path"] = source if source.is_absolute() else base / source
                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        data = np.load(self.records[index]["path"])
        item = {
            "points": np.asarray(data["points"], dtype=np.float32),
            "colors": np.asarray(data["colors"], dtype=np.float32),
        }
        if "labels" in data:
            item["labels"] = np.asarray(data["labels"], dtype=np.int64)
        return item


def sparse_sample(points: np.ndarray, colors: np.ndarray, voxel_size_m: float, labels=None) -> dict:
    coordinates, features, _inverse = voxelize(points, colors, voxel_size_m)
    result = {"coordinates": coordinates, "features": features}
    if labels is not None:
        first = np.unique(np.floor(points / voxel_size_m).astype(np.int32), axis=0, return_index=True)[1]
        result["labels"] = np.asarray(labels, dtype=np.int64)[first]
    return result
