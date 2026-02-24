from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _stack_sparse_coords(coords_list: List[np.ndarray]) -> torch.Tensor:
    stacked = []
    for batch_idx, coords in enumerate(coords_list):
        batch_column = np.full((coords.shape[0], 1), batch_idx, dtype=np.int32)
        stacked.append(np.concatenate([batch_column, coords.astype(np.int32)], axis=1))
    return torch.from_numpy(np.concatenate(stacked, axis=0)).int()


def _stack_features(feats_list: List[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(feats_list, axis=0)).float()


def collate_contrastive(batch: List[Dict]) -> Dict:
    coords_a = _stack_sparse_coords([sample["coords_a"] for sample in batch])
    feats_a = _stack_features([sample["features_a"] for sample in batch])
    coords_b = _stack_sparse_coords([sample["coords_b"] for sample in batch])
    feats_b = _stack_features([sample["features_b"] for sample in batch])
    paths = [sample["pcd_path"] for sample in batch]
    return {
        "coords_a": coords_a,
        "features_a": feats_a,
        "coords_b": coords_b,
        "features_b": feats_b,
        "pcd_paths": paths,
    }


def collate_supervised(batch: List[Dict]) -> Dict:
    coords = _stack_sparse_coords([sample["coords"] for sample in batch])
    feats = _stack_features([sample["features"] for sample in batch])
    labels = torch.from_numpy(np.concatenate([sample["labels"] for sample in batch], axis=0)).long()
    sample_ids = [sample["sample_id"] for sample in batch]
    return {
        "coords": coords,
        "features": feats,
        "labels": labels,
        "sample_ids": sample_ids,
    }

