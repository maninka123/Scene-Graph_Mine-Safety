from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from MinkUNET.minkunet.data.pcd import read_pcd, write_colored_pcd

from func_segment_minkunet import MinkUNETObjectSegmenter


def _extract_timestamp(filepath: str) -> str:
    filename = os.path.basename(filepath)
    match = re.search(r"PC_(\d{8}_\d+)\.pcd", filename)
    return match.group(1) if match else filename


def segment_geometry_temporal_minkunet(
    dataset_folder: str,
    output_dir: str,
    config_path: str = "MinkUNET/configs/inference.yaml",
    checkpoint_path: Optional[str] = None,
    temporal_window: int = 3,
    visualize: bool = False,
    class_dbscan_eps: Optional[Dict[str, float]] = None,
    class_min_points: Optional[Dict[str, int]] = None,
) -> Optional[list[str]]:
    """
    Process a sequence of PCD files with MinkUNET segmentation and export
    per-frame object JSONs compatible with temporal scene graph builder.
    """
    print(f"\n{'='*50}")
    print("   [Function] Temporal MinkUNET Segmentation")
    print(f"{'='*50}\n")

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pcd_pattern = os.path.join(dataset_folder, "PC_*.pcd")
    pcd_files = sorted(glob.glob(pcd_pattern))
    if not pcd_files:
        print(f"[ERROR] No PCD files found matching: {pcd_pattern}")
        return None

    print(f"[STEP 1] Found {len(pcd_files)} frames in {dataset_folder}")

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

    frame_json_files: list[str] = []
    frame_metadata = []
    last_pred_visual = None

    for frame_idx, pcd_path in enumerate(pcd_files):
        timestamp = _extract_timestamp(pcd_path)
        print(f"\n[FRAME {frame_idx:02d}] {os.path.basename(pcd_path)}")

        points, colors = read_pcd(pcd_path)
        pred = segmenter.infer_labels(points=points, colors=colors)
        labels = pred["labels"]
        pred_colors = pred["pred_colors"]

        objects = segmenter.objects_from_labels(
            points=points,
            labels=labels,
            frame_id=frame_idx,
            timestamp=timestamp,
        )

        out_json = os.path.join(frames_dir, f"frame_{frame_idx:02d}_objects.json")
        out_seg_pcd = os.path.join(frames_dir, f"frame_{frame_idx:02d}_semantic.pcd")
        out_pred_npz = os.path.join(frames_dir, f"frame_{frame_idx:02d}_prediction.npz")

        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(objects, handle, indent=2)
        write_colored_pcd(out_seg_pcd, points, pred_colors)
        np.savez_compressed(
            out_pred_npz,
            points=points.astype(np.float32),
            colors=colors.astype(np.float32),
            labels=labels.astype(np.int64),
            class_names=np.array(segmenter.class_names),
        )

        frame_json_files.append(out_json)
        frame_metadata.append(
            {
                "frame_id": int(frame_idx),
                "timestamp": timestamp,
                "source_file": str(Path(pcd_path).resolve()),
                "num_objects": int(len(objects)),
                "objects_file": str(Path(out_json).resolve()),
                "semantic_pcd": str(Path(out_seg_pcd).resolve()),
            }
        )
        print(f"         Objects: {len(objects)}")
        last_pred_visual = (points, pred_colors)

    frame_index = {
        "total_frames": int(len(frame_metadata)),
        "frames": frame_metadata,
    }
    index_file = os.path.join(output_dir, "frame_index.json")
    with open(index_file, "w", encoding="utf-8") as handle:
        json.dump(frame_index, handle, indent=2)

    print(f"\n[SUCCESS] Processed {len(frame_json_files)} frames.")
    print(f"[SUCCESS] Frame index saved: {index_file}")

    if visualize and last_pred_visual is not None:
        import open3d as o3d

        points, pred_colors = last_pred_visual
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(pred_colors.astype(np.float64))
        o3d.visualization.draw_geometries([cloud], window_name="Last Frame Semantic Segmentation")

    return frame_json_files

