from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _rotation_matrix_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def augment_points(
    points: np.ndarray,
    colors: np.ndarray,
    labels: Optional[np.ndarray],
    aug_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not aug_cfg:
        return points, colors, labels

    pts = points.copy()
    cols = colors.copy()
    lbl = labels.copy() if labels is not None else None

    rot_deg = float(aug_cfg.get("rotation_deg", 0.0))
    if rot_deg > 0:
        theta = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
        rot = _rotation_matrix_z(theta)
        pts = pts @ rot.T

    scale_min = float(aug_cfg.get("scale_min", 1.0))
    scale_max = float(aug_cfg.get("scale_max", 1.0))
    if scale_max > 0 and (scale_min != 1.0 or scale_max != 1.0):
        scale = np.random.uniform(scale_min, scale_max)
        pts *= scale

    jitter_std = float(aug_cfg.get("jitter_std", 0.0))
    jitter_clip = float(aug_cfg.get("jitter_clip", 0.0))
    if jitter_std > 0:
        jitter = np.random.normal(0.0, jitter_std, size=pts.shape).astype(np.float32)
        if jitter_clip > 0:
            jitter = np.clip(jitter, -jitter_clip, jitter_clip)
        pts += jitter

    translation = float(aug_cfg.get("translation", 0.0))
    if translation > 0:
        shift = np.random.uniform(-translation, translation, size=(1, 3)).astype(np.float32)
        pts += shift

    dropout = float(aug_cfg.get("point_dropout", 0.0))
    if 0 < dropout < 1 and pts.shape[0] > 64:
        keep_mask = np.random.rand(pts.shape[0]) > dropout
        if keep_mask.sum() < 64:
            keep_indices = np.random.choice(pts.shape[0], size=64, replace=False)
            keep_mask[keep_indices] = True
        pts = pts[keep_mask]
        cols = cols[keep_mask]
        if lbl is not None:
            lbl = lbl[keep_mask]

    return pts, cols, lbl

