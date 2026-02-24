from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _common import abs_path, load_cfg
from minkunet.data.pcd import read_pcd, write_colored_pcd
from minkunet.engine.checkpoint import load_checkpoint
from minkunet.engine.infer import TemporalMajoritySmoother, infer_single_frame, labels_to_colors
from minkunet.models.network import SemanticMinkUNet
from minkunet.utils.io import ensure_dir


def _collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pcd":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.pcd"))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on PCD data.")
    parser.add_argument("--config", default="MinkUNET/configs/inference.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", default=None, help="PCD file or folder containing PCD files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temporal-window", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    inference_cfg = cfg.get("inference", {})

    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_raw = args.checkpoint or inference_cfg.get("checkpoint") or "MinkUNET/checkpoints/semantic_best.pt"
    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.is_absolute() and len(checkpoint_path.parts) == 1:
        checkpoint_path = Path(cfg["checkpoint"]["dir"]) / checkpoint_path
    checkpoint_path = abs_path(checkpoint_path)

    input_path = args.input or inference_cfg.get("input_path") or "Datasets/Data_all"
    input_path = abs_path(input_path)
    output_dir = ensure_dir(abs_path(args.output_dir or inference_cfg.get("output_dir") or "Results/MinkUNET_inference"))
    temporal_window = int(args.temporal_window or inference_cfg.get("temporal_window", 1))

    pcd_files = _collect_inputs(input_path)
    if not pcd_files:
        raise FileNotFoundError(f"No PCD inputs found for: {input_path}")

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

    temporal_smoother = TemporalMajoritySmoother(window=temporal_window) if temporal_window > 1 else None

    summary = {"frames": []}
    class_names = cfg["model"]["class_names"]
    class_colors = cfg["model"]["class_colors"]

    for pcd_path in tqdm(pcd_files, desc="Inference"):
        points, colors = read_pcd(pcd_path)
        results = infer_single_frame(
            model=model,
            points=points,
            colors=colors,
            cfg=cfg,
            device=device,
            temporal_smoother=temporal_smoother,
        )

        full_labels = results["point_predictions_full"].astype(np.int64)
        pred_colors = labels_to_colors(full_labels, class_names=class_names, class_colors=class_colors)

        stem = pcd_path.stem
        frame_dir = ensure_dir(output_dir / stem)
        npz_out = frame_dir / "prediction.npz"
        pcd_out = frame_dir / "prediction.pcd"
        json_out = frame_dir / "summary.json"

        np.savez_compressed(
            npz_out,
            points=points.astype(np.float32),
            colors=colors.astype(np.float32),
            labels=full_labels,
            class_names=np.array(class_names),
        )
        write_colored_pcd(pcd_out, points, pred_colors)

        counts = {}
        for class_id, class_name in enumerate(class_names):
            counts[class_name] = int((full_labels == class_id).sum())
        counts["unlabeled"] = int((full_labels < 0).sum())

        frame_summary = {
            "source_pcd": str(pcd_path.resolve()),
            "frame_dir": str(frame_dir.resolve()),
            "prediction_npz": str(npz_out.resolve()),
            "prediction_pcd": str(pcd_out.resolve()),
            "point_count": int(points.shape[0]),
            "class_counts": counts,
        }
        with json_out.open("w", encoding="utf-8") as handle:
            json.dump(frame_summary, handle, indent=2)
        summary["frames"].append(frame_summary)

    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[Done] Inference outputs written to {output_dir}")


if __name__ == "__main__":
    main()
