from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from _common import abs_path, load_cfg
from minkunet.data.pcd import read_pcd, write_colored_pcd
from minkunet.engine.checkpoint import load_checkpoint
from minkunet.engine.infer import TemporalMajoritySmoother, infer_single_frame, labels_to_colors
from minkunet.models.network import SemanticMinkUNet
from minkunet.utils.io import ensure_dir

# User-editable defaults (can be overridden by CLI args)
DEFAULT_CONFIG = "MinkUNET/configs/inference.yaml"
DEFAULT_CHECKPOINT = "MinkUNET/checkpoints/semantic_best.pt"
DEFAULT_INPUT_PCD = "Datasets/Data_all"
DEFAULT_OUTPUT_DIR = "Results/MinkUNET_preview"
DEFAULT_TEMPORAL_WINDOW = 1
DEFAULT_SHOW_WINDOW = True


def _rgb_uint8(class_colors: dict, class_name: str) -> tuple[int, int, int]:
    color = class_colors.get(class_name, [255, 255, 255])
    if len(color) != 3:
        return 255, 255, 255
    return int(color[0]), int(color[1]), int(color[2])


def _resolve_input_pcd(input_value: str) -> Path:
    p = abs_path(input_value)
    if p.is_file() and p.suffix.lower() == ".pcd":
        return p
    if p.is_dir():
        files = sorted(p.glob("*.pcd"))
        if not files:
            raise FileNotFoundError(f"No .pcd files in folder: {p}")
        return files[0]
    raise FileNotFoundError(f"Input path is not a .pcd file or folder: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview semantic segmentation on a single point cloud.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", default=DEFAULT_INPUT_PCD, help="Single .pcd file or folder (uses first file).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temporal-window", type=int, default=DEFAULT_TEMPORAL_WINDOW)
    parser.add_argument("--no-view", action="store_true", help="Do not open Open3D visualization window.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = abs_path(args.checkpoint)
    input_pcd = _resolve_input_pcd(args.input)
    output_dir = ensure_dir(abs_path(args.output_dir))

    try:
        model = SemanticMinkUNet(
            in_channels=int(cfg["model"]["in_channels"]),
            feature_dim=int(cfg["model"]["feature_dim"]),
            num_classes=int(cfg["model"]["num_classes"]),
            width=int(cfg["model"]["width"]),
        ).to(device)
    except ImportError as exc:
        print(str(exc))
        sys.exit(1)

    load_checkpoint(checkpoint_path, model=model, map_location=str(device))
    model.eval()

    points, colors = read_pcd(input_pcd)
    temporal_window = int(max(1, args.temporal_window))
    temporal_smoother = TemporalMajoritySmoother(window=temporal_window) if temporal_window > 1 else None

    results = infer_single_frame(
        model=model,
        points=points,
        colors=colors,
        cfg=cfg,
        device=device,
        temporal_smoother=temporal_smoother,
    )
    labels = results["point_predictions_full"].astype(np.int64)
    pred_colors = labels_to_colors(labels, class_names=cfg["model"]["class_names"], class_colors=cfg["model"]["class_colors"])

    stem = input_pcd.stem
    out_npz = output_dir / f"{stem}_preview_prediction.npz"
    out_pcd = output_dir / f"{stem}_preview_prediction.pcd"
    out_json = output_dir / f"{stem}_preview_summary.json"

    np.savez_compressed(
        out_npz,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        labels=labels.astype(np.int64),
        class_names=np.array(cfg["model"]["class_names"]),
    )
    write_colored_pcd(out_pcd, points, pred_colors)

    class_counts = {
        class_name: int((labels == class_id).sum())
        for class_id, class_name in enumerate(cfg["model"]["class_names"])
    }
    class_counts["unlabeled"] = int((labels < 0).sum())

    legend_lines = ["class_id,class_name,r,g,b,count"]
    for class_id, class_name in enumerate(cfg["model"]["class_names"]):
        r, g, b = _rgb_uint8(cfg["model"]["class_colors"], class_name)
        count = class_counts.get(class_name, 0)
        legend_lines.append(f"{class_id},{class_name},{r},{g},{b},{count}")
    legend_lines.append(f"-1,unlabeled,0,0,0,{class_counts['unlabeled']}")

    payload = {
        "input_pcd": str(input_pcd.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "prediction_npz": str(out_npz.resolve()),
        "prediction_pcd": str(out_pcd.resolve()),
        "point_count": int(points.shape[0]),
        "class_counts": class_counts,
    }
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    legend_txt = output_dir / f"{stem}_label_legend.txt"
    legend_txt.write_text("\n".join(legend_lines) + "\n", encoding="utf-8")

    print(f"[Done] Prediction summary: {out_json}")
    print(f"[Done] Colored prediction PCD: {out_pcd}")
    print(f"[Done] Label legend: {legend_txt}")
    print("[Legend] class_id class_name rgb count")
    for line in legend_lines[1:]:
        cid, cname, r, g, b, count = line.split(",")
        print(f"[Legend] {cid:>2} {cname:<10} ({r},{g},{b}) count={count}")

    show_window = DEFAULT_SHOW_WINDOW and (not args.no_view)
    if show_window:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(pred_colors.astype(np.float64))
        o3d.visualization.draw_geometries(
            [cloud],
            window_name=f"Prediction Preview: {input_pcd.name}",
            width=1400,
            height=900,
        )


if __name__ == "__main__":
    main()
