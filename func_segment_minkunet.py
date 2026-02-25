from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import torch

from MinkUNET.minkunet.config import load_config
from MinkUNET.minkunet.data.pcd import read_pcd, write_colored_pcd
from MinkUNET.minkunet.engine.checkpoint import load_checkpoint
from MinkUNET.minkunet.engine.infer import TemporalMajoritySmoother, infer_single_frame, labels_to_colors
from MinkUNET.minkunet.models.network import SemanticMinkUNet


DEFAULT_DBSCAN_EPS = {
    "wall": 0.60,
    "roof": 0.60,
    "equipment": 0.35,
    "human": 0.25,
    "conveyor": 0.40,
    "other": 0.40,
}

DEFAULT_MIN_POINTS = {
    "wall": 300,
    "roof": 300,
    "equipment": 80,
    "human": 20,
    "conveyor": 120,
    "other": 80,
}


def _resolve_checkpoint_path(cfg: Dict, checkpoint_path: Optional[str]) -> Path:
    candidates = []
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))

    inference_cfg = cfg.get("inference", {})
    if inference_cfg.get("checkpoint"):
        candidates.append(Path(str(inference_cfg["checkpoint"])))

    candidates.append(Path("MinkUNET/checkpoints/semantic_best.pt"))

    resolved_candidates = []
    for cand in candidates:
        if cand.is_absolute():
            resolved_candidates.append(cand)
        else:
            resolved_candidates.append((Path.cwd() / cand).resolve())
            if len(cand.parts) == 1:
                ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "MinkUNET/checkpoints"))
                resolved_candidates.append((Path.cwd() / ckpt_dir / cand).resolve())

    for cand in resolved_candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "Could not find semantic checkpoint. Tried:\n"
        + "\n".join(f"  - {p}" for p in resolved_candidates)
    )


class MinkUNETObjectSegmenter:
    """
    Runs MinkUNET semantic segmentation and converts semantic labels into object instances
    via class-aware DBSCAN clustering.
    """

    def __init__(
        self,
        config_path: str = "MinkUNET/configs/inference.yaml",
        checkpoint_path: Optional[str] = None,
        temporal_window: int = 1,
        class_dbscan_eps: Optional[Dict[str, float]] = None,
        class_min_points: Optional[Dict[str, int]] = None,
    ) -> None:
        self.cfg = load_config(config_path)
        device_str = str(self.cfg.get("device", "cuda"))
        if device_str == "cuda" and not torch.cuda.is_available():
            print("[Warn] CUDA requested but not available, falling back to CPU.")
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.checkpoint_path = _resolve_checkpoint_path(self.cfg, checkpoint_path=checkpoint_path)
        self.class_names = [str(n) for n in self.cfg["model"]["class_names"]]
        self.class_colors = self.cfg["model"]["class_colors"]

        self.dbscan_eps = DEFAULT_DBSCAN_EPS.copy()
        self.min_points = DEFAULT_MIN_POINTS.copy()
        if class_dbscan_eps:
            self.dbscan_eps.update({str(k): float(v) for k, v in class_dbscan_eps.items()})
        if class_min_points:
            self.min_points.update({str(k): int(v) for k, v in class_min_points.items()})

        self.model = SemanticMinkUNet(
            in_channels=int(self.cfg["model"]["in_channels"]),
            feature_dim=int(self.cfg["model"]["feature_dim"]),
            num_classes=int(self.cfg["model"]["num_classes"]),
            width=int(self.cfg["model"]["width"]),
        ).to(self.device)
        load_checkpoint(self.checkpoint_path, model=self.model, map_location=str(self.device))
        self.model.eval()

        self.temporal_smoother = TemporalMajoritySmoother(window=max(1, int(temporal_window))) if temporal_window > 1 else None

    def infer_labels(self, points: np.ndarray, colors: np.ndarray) -> Dict:
        infer_out = infer_single_frame(
            model=self.model,
            points=points,
            colors=colors,
            cfg=self.cfg,
            device=self.device,
            temporal_smoother=self.temporal_smoother,
        )
        labels = infer_out["point_predictions_full"].astype(np.int64)
        pred_colors = labels_to_colors(labels, class_names=self.class_names, class_colors=self.class_colors)
        return {
            "labels": labels,
            "pred_colors": pred_colors,
            "infer_out": infer_out,
        }

    def objects_from_labels(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> list[Dict]:
        objects: list[Dict] = []
        next_id = 0

        for class_id, class_name in enumerate(self.class_names):
            class_mask = labels == class_id
            class_points = points[class_mask]
            if class_points.shape[0] == 0:
                continue

            eps = float(self.dbscan_eps.get(class_name, 0.4))
            min_pts = int(self.min_points.get(class_name, 60))

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(class_points.astype(np.float64))
            cluster_ids = np.array(
                cloud.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False),
                dtype=np.int64,
            )

            unique_clusters = [int(c) for c in np.unique(cluster_ids).tolist() if c >= 0]
            # Fallback: keep small/merged class blob as one object for key dynamic classes.
            if not unique_clusters and class_name in {"equipment", "human", "conveyor"}:
                unique_clusters = [0]
                cluster_ids = np.zeros((class_points.shape[0],), dtype=np.int64)

            for cid in unique_clusters:
                pts = class_points[cluster_ids == cid]
                if pts.shape[0] == 0:
                    continue
                min_bound = pts.min(axis=0)
                max_bound = pts.max(axis=0)
                center = ((min_bound + max_bound) / 2.0).tolist()
                dims = (max_bound - min_bound).tolist()
                volume = float(np.prod(np.maximum(max_bound - min_bound, 0.0)))

                entry = {
                    "id": int(next_id),
                    "label": class_name,
                    "confidence": 1.0,
                    "centroid": [float(x) for x in center],
                    "dimensions": [float(x) for x in dims],
                    "volume": float(volume),
                    "point_count": int(pts.shape[0]),
                    "class_id": int(class_id),
                }
                if frame_id is not None:
                    entry["frame_id"] = int(frame_id)
                if timestamp is not None:
                    entry["timestamp"] = str(timestamp)

                objects.append(entry)
                next_id += 1

        return objects


def segment_geometry_minkunet(
    pcd_path: str,
    output_dir: str,
    config_path: str = "MinkUNET/configs/inference.yaml",
    checkpoint_path: Optional[str] = None,
    temporal_window: int = 1,
    visualize: bool = False,
    class_dbscan_eps: Optional[Dict[str, float]] = None,
    class_min_points: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """
    Segment a single frame using MinkUNET and export scene objects JSON
    compatible with scene-graph builders.
    """
    print(f"\n{'='*50}")
    print("   [Function] MinkUNET Semantic Segmentation")
    print(f"{'='*50}\n")

    if not os.path.exists(pcd_path):
        print(f"[ERROR] File not found: {pcd_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_objects_json = os.path.join(output_dir, "scene_objects_segmented.json")
    out_pred_pcd = os.path.join(output_dir, "semantic_segmentation.pcd")
    out_pred_npz = os.path.join(output_dir, "semantic_prediction.npz")
    out_summary_json = os.path.join(output_dir, "semantic_summary.json")

    try:
        segmenter = MinkUNETObjectSegmenter(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            temporal_window=temporal_window,
            class_dbscan_eps=class_dbscan_eps,
            class_min_points=class_min_points,
        )
    except Exception as exc:
        print(f"[ERROR] Could not initialize MinkUNET segmenter: {exc}")
        return None

    points, colors = read_pcd(pcd_path)
    pred = segmenter.infer_labels(points=points, colors=colors)
    labels = pred["labels"]
    pred_colors = pred["pred_colors"]

    objects = segmenter.objects_from_labels(points=points, labels=labels)
    if not objects:
        print("[WARN] No objects extracted from semantic labels.")

    with open(out_objects_json, "w", encoding="utf-8") as handle:
        json.dump(objects, handle, indent=2)

    write_colored_pcd(out_pred_pcd, points, pred_colors)
    np.savez_compressed(
        out_pred_npz,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        labels=labels.astype(np.int64),
        class_names=np.array(segmenter.class_names),
    )

    counts = {name: int((labels == idx).sum()) for idx, name in enumerate(segmenter.class_names)}
    counts["unlabeled"] = int((labels < 0).sum())
    summary = {
        "source_pcd": str(Path(pcd_path).resolve()),
        "checkpoint": str(segmenter.checkpoint_path.resolve()),
        "objects_json": str(Path(out_objects_json).resolve()),
        "semantic_pcd": str(Path(out_pred_pcd).resolve()),
        "semantic_npz": str(Path(out_pred_npz).resolve()),
        "point_count": int(points.shape[0]),
        "object_count": int(len(objects)),
        "class_counts": counts,
    }
    with open(out_summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[SUCCESS] Saved segmented objects to: {out_objects_json}")
    print(f"[SUCCESS] Saved semantic colored PCD to: {out_pred_pcd}")

    if visualize:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(pred_colors.astype(np.float64))
        o3d.visualization.draw_geometries([cloud], window_name="MinkUNET Semantic Segmentation")

    return out_objects_json

