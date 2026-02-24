from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _common import abs_path, load_cfg
from minkunet.data.pcd import read_pcd, write_colored_pcd
from minkunet.utils.io import ensure_dir, write_json


def _roi_mask(points: np.ndarray, roi_cfg: dict) -> np.ndarray:
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


def _cluster_colors(labels: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for i, lbl in enumerate(unique):
        if lbl < 0:
            colors[labels == lbl] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            colors[labels == lbl] = np.array(cmap(i % 20)[:3], dtype=np.float32)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cluster proposals for manual annotation.")
    parser.add_argument("--config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--eps", type=float, default=0.35)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    class_names = cfg["model"]["class_names"]

    sample_dir = abs_path(data_cfg["sample_dir"])
    output_dir = ensure_dir(abs_path(data_cfg["cluster_proposals_dir"]))
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    pcd_files = sorted(sample_dir.glob("*.pcd"))
    if not pcd_files:
        raise RuntimeError(
            f"No sample PCD files in {sample_dir}. Run create_annotation_sample.py first."
        )

    for pcd_path in pcd_files:
        stem = pcd_path.stem
        proposal_npz = output_dir / f"{stem}_cluster_proposal.npz"
        label_template_path = output_dir / f"{stem}_cluster_labels.json"
        preview_pcd_path = output_dir / f"{stem}_clusters_preview.pcd"

        if proposal_npz.exists() and label_template_path.exists() and not args.overwrite:
            print(f"[Skip] {stem} already has proposal files")
            continue

        points, colors = read_pcd(pcd_path)
        roi_mask = _roi_mask(points, data_cfg["roi"])
        points_roi = points[roi_mask]
        colors_roi = colors[roi_mask]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_roi.astype(np.float64))
        labels = np.array(
            cloud.cluster_dbscan(
                eps=float(args.eps),
                min_points=int(args.min_points),
                print_progress=False,
            ),
            dtype=np.int64,
        )

        np.savez_compressed(
            proposal_npz,
            points=points_roi.astype(np.float32),
            colors=colors_roi.astype(np.float32),
            cluster_ids=labels,
            source_pcd=str(pcd_path.resolve()),
        )

        cluster_preview_colors = _cluster_colors(labels)
        write_colored_pcd(preview_pcd_path, points_roi, cluster_preview_colors)

        clusters = []
        for cluster_id in sorted(np.unique(labels).tolist()):
            point_count = int((labels == cluster_id).sum())
            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "point_count": point_count,
                    "manual_label": "",
                    "notes": "",
                }
            )

        payload = {
            "source_pcd": str(pcd_path.resolve()),
            "proposal_npz": str(proposal_npz.resolve()),
            "preview_pcd": str(preview_pcd_path.resolve()),
            "allowed_labels": class_names,
            "instructions": "Fill manual_label for each cluster_id. Leave blank to ignore that cluster.",
            "clusters": clusters,
        }
        write_json(label_template_path, payload)

        print(
            f"[Cluster] {stem}: points={points_roi.shape[0]} clusters={len(clusters)} "
            f"template={label_template_path.name}"
        )

    print(f"[Done] Cluster proposals written to {output_dir}")


if __name__ == "__main__":
    main()

