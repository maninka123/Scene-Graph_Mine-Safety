from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, Optional

import numpy as np
import torch

from ..data.preprocess import preprocess_sample
from ..models.network import sparse_tensor_from_batch


class TemporalMajoritySmoother:
    def __init__(self, window: int = 3):
        self.window = max(1, int(window))
        self.history = defaultdict(lambda: deque(maxlen=self.window))

    def smooth(self, voxel_coords: np.ndarray, voxel_labels: np.ndarray) -> np.ndarray:
        if self.window <= 1:
            return voxel_labels
        smoothed = voxel_labels.copy()
        for idx, coord in enumerate(voxel_coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            self.history[key].append(int(voxel_labels[idx]))
            smoothed[idx] = Counter(self.history[key]).most_common(1)[0][0]
        return smoothed


@torch.no_grad()
def infer_single_frame(
    model,
    points: np.ndarray,
    colors: np.ndarray,
    cfg: Dict,
    device: torch.device,
    temporal_smoother: Optional[TemporalMajoritySmoother] = None,
) -> Dict:
    data_cfg = cfg["data"]
    processed = preprocess_sample(
        points=points,
        colors=colors,
        labels=None,
        data_cfg=data_cfg,
        max_points=int(data_cfg["max_points_eval"]),
        augmentation_cfg=None,
    )

    voxel_coords = processed["coords"]
    voxel_features = processed["features"]

    batch_col = np.zeros((voxel_coords.shape[0], 1), dtype=np.int32)
    batched_coords = np.concatenate([batch_col, voxel_coords], axis=1)
    coords_tensor = torch.from_numpy(batched_coords).int()
    feats_tensor = torch.from_numpy(voxel_features).float()

    sparse_input = sparse_tensor_from_batch(coords_tensor, feats_tensor, device=device)
    logits = model(sparse_input).F
    voxel_predictions = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

    if temporal_smoother is not None:
        voxel_predictions = temporal_smoother.smooth(voxel_coords, voxel_predictions)

    point_predictions = voxel_predictions[processed["inverse_indices"]]

    full_predictions = np.full(processed["original_count"], -1, dtype=np.int64)
    full_predictions[processed["point_indices"]] = point_predictions

    return {
        "processed_points": processed["points"],
        "processed_colors": processed["colors"],
        "voxel_coords": voxel_coords,
        "voxel_predictions": voxel_predictions,
        "point_predictions_processed": point_predictions,
        "point_predictions_full": full_predictions,
    }


def labels_to_colors(labels: np.ndarray, class_names, class_colors: Dict[str, list]) -> np.ndarray:
    color_lut = {}
    for class_id, class_name in enumerate(class_names):
        color = class_colors.get(class_name, [255, 255, 255])
        color_lut[class_id] = np.array(color, dtype=np.float32) / 255.0

    output = np.zeros((labels.shape[0], 3), dtype=np.float32)
    valid_mask = labels >= 0
    for class_id, color in color_lut.items():
        output[(labels == class_id) & valid_mask] = color
    output[~valid_mask] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return output

