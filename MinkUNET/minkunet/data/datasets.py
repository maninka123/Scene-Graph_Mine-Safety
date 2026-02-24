from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from ..utils.io import read_jsonl, resolve_path
from .pcd import read_pcd
from .preprocess import preprocess_sample


class UnlabeledContrastiveDataset(Dataset):
    def __init__(self, manifest_path: str | Path, cfg: Dict, augmentation_cfg: Dict):
        self.records = read_jsonl(manifest_path)
        self.cfg = cfg
        self.augmentation_cfg = augmentation_cfg
        if not self.records:
            raise ValueError(f"No records found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        pcd_path = resolve_path(record["pcd_path"])
        points, colors = read_pcd(pcd_path)

        view_a = preprocess_sample(
            points=points,
            colors=colors,
            labels=None,
            data_cfg=self.cfg["data"],
            max_points=int(self.cfg["data"]["max_points_train"]),
            augmentation_cfg=self.augmentation_cfg,
        )
        view_b = preprocess_sample(
            points=points,
            colors=colors,
            labels=None,
            data_cfg=self.cfg["data"],
            max_points=int(self.cfg["data"]["max_points_train"]),
            augmentation_cfg=self.augmentation_cfg,
        )

        return {
            "coords_a": view_a["coords"],
            "features_a": view_a["features"],
            "coords_b": view_b["coords"],
            "features_b": view_b["features"],
            "pcd_path": str(pcd_path),
        }


class LabeledNPZDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: Dict,
        train: bool,
        augmentation_cfg: Dict | None = None,
    ):
        self.records = read_jsonl(manifest_path)
        self.cfg = cfg
        self.train = train
        self.augmentation_cfg = augmentation_cfg if train else None
        if not self.records:
            raise ValueError(f"No records found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        npz_path = resolve_path(record["label_npz"])
        payload = np.load(npz_path)

        points = payload["points"].astype(np.float32)
        colors = payload["colors"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)

        processed = preprocess_sample(
            points=points,
            colors=colors,
            labels=labels,
            data_cfg=self.cfg["data"],
            max_points=int(
                self.cfg["data"]["max_points_train"] if self.train else self.cfg["data"]["max_points_eval"]
            ),
            augmentation_cfg=self.augmentation_cfg,
        )

        voxel_labels = processed["voxel_labels"]
        if voxel_labels is None:
            raise ValueError(f"Missing voxel labels for sample: {npz_path}")

        return {
            "coords": processed["coords"],
            "features": processed["features"],
            "labels": voxel_labels.astype(np.int64),
            "sample_id": npz_path.stem,
        }


def list_manifest_paths(manifests_dir: str | Path) -> List[Path]:
    path = resolve_path(manifests_dir)
    return sorted(path.glob("*.jsonl"))

