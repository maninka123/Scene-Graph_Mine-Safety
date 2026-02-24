from __future__ import annotations

from typing import Dict, List

import numpy as np


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    pred = prediction.reshape(-1)
    tgt = target.reshape(-1)
    valid = tgt != ignore_index
    pred = pred[valid]
    tgt = tgt[valid]
    if pred.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    bins = num_classes * tgt + pred
    hist = np.bincount(bins, minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)


def compute_segmentation_metrics(hist: np.ndarray, class_names: List[str]) -> Dict:
    epsilon = 1e-8
    diagonal = np.diag(hist).astype(np.float64)
    per_class_denom = hist.sum(1) + hist.sum(0) - diagonal
    per_class_iou = diagonal / (per_class_denom + epsilon)

    pixel_accuracy = diagonal.sum() / (hist.sum() + epsilon)
    mean_iou = float(np.nanmean(per_class_iou))

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "iou": float(per_class_iou[idx]),
            "tp": int(diagonal[idx]),
            "gt_points": int(hist[idx, :].sum()),
            "pred_points": int(hist[:, idx].sum()),
        }

    return {
        "miou": mean_iou,
        "accuracy": float(pixel_accuracy),
        "per_class": per_class,
    }

