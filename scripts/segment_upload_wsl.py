from __future__ import annotations

import argparse
import json
from importlib import util
from pathlib import Path

import numpy as np
import torch

CLASS_NAMES = ["wall", "equipment", "human", "conveyor", "roof", "other"]
ROI_MIN = np.array([-10.0, -10.0, -3.0], dtype=np.float32)
ROI_MAX = np.array([10.0, 10.0, 3.0], dtype=np.float32)
ROI_SCALE = np.array([10.0, 10.0, 3.0], dtype=np.float32)
VOXEL_SIZE_M = 0.05
MAX_POINTS = 200_000


def progress(stage: str, fraction: float, message: str) -> None:
    print(
        "MINEGRAPH_PROGRESS "
        + json.dumps({"stage": stage, "progress": fraction, "message": message}),
        flush=True,
    )


def load_network_module():
    path = Path(__file__).resolve().parents[1] / "src" / "mine_safety" / "perception" / "network.py"
    specification = util.spec_from_file_location("minegraph_sparse_network", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load sparse network module from {path}")
    module = util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data.get("colors", np.zeros_like(points)), dtype=np.float32)
    else:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Point cloud must contain a non-empty [N,3] XYZ array")
    if colors.shape != points.shape:
        colors = np.zeros_like(points)
    if colors.size and colors.max() > 1.0:
        colors /= 255.0
    finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    return points[finite], np.clip(colors[finite], 0.0, 1.0)


def preprocess(points: np.ndarray, colors: np.ndarray) -> dict[str, np.ndarray | int]:
    original_count = len(points)
    original_indices = np.arange(original_count, dtype=np.int64)
    mask = ((points >= ROI_MIN) & (points <= ROI_MAX)).all(axis=1)
    selected_points = points[mask]
    selected_colors = colors[mask]
    selected_indices = original_indices[mask]
    if not len(selected_points):
        selected_points, selected_colors, selected_indices = points, colors, original_indices
    if len(selected_points) > MAX_POINTS:
        keep = np.linspace(0, len(selected_points) - 1, MAX_POINTS, dtype=np.int64)
        selected_points = selected_points[keep]
        selected_colors = selected_colors[keep]
        selected_indices = selected_indices[keep]

    features = np.concatenate(
        [selected_points / ROI_SCALE[None, :], selected_colors], axis=1
    ).astype(np.float32)
    coordinates = np.floor(selected_points / VOXEL_SIZE_M).astype(np.int32)
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.zeros((len(unique), features.shape[1]), dtype=np.float64)
    for column in range(features.shape[1]):
        np.add.at(sums[:, column], inverse, features[:, column])
    voxel_features = (sums / counts[:, None]).astype(np.float32)
    return {
        "coordinates": unique,
        "features": voxel_features,
        "inverse": inverse,
        "point_indices": selected_indices,
        "original_count": original_count,
    }


def legacy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state.items():
        key = key.replace("backbone.out_proj.", "backbone.output.")
        key = key.replace(".merge.", ".refine.")
        key = key.replace("classifier.", "head.")
        mapped[key] = value
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    progress("reading", 0.12, "Reading and validating XYZ/RGB points")
    points, colors = read_cloud(args.input)
    progress("voxelise", 0.25, f"Voxelising {len(points):,} points at 5 cm")
    processed = preprocess(points, colors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress("model", 0.38, f"Loading six-class MinkUNet on {device.type.upper()}")
    network = load_network_module()
    SemanticMinkUNet = network.SemanticMinkUNet
    model = SemanticMinkUNet(
        in_channels=6, width=32, feature_dim=96, num_classes=len(CLASS_NAMES)
    ).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(legacy_state(checkpoint["model_state"]), strict=True)
    model.eval()

    progress("minkunet", 0.52, "Running sparse 3D semantic segmentation")
    me = network.require_minkowski()
    coordinates = processed["coordinates"]
    batch = np.zeros((len(coordinates), 1), dtype=np.int32)
    sparse = me.SparseTensor(
        coordinates=torch.from_numpy(np.concatenate([batch, coordinates], axis=1)).int().to(device),
        features=torch.from_numpy(processed["features"]).float().to(device),
    )
    with torch.inference_mode():
        logits = model(sparse).F
        voxel_probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    inverse = processed["inverse"]
    point_probabilities = voxel_probabilities[inverse]
    point_labels = point_probabilities.argmax(axis=1).astype(np.int64)
    full_labels = np.full(processed["original_count"], -1, dtype=np.int64)
    full_probabilities = np.zeros(
        (processed["original_count"], len(CLASS_NAMES)), dtype=np.float32
    )
    full_labels[processed["point_indices"]] = point_labels
    full_probabilities[processed["point_indices"]] = point_probabilities

    progress("export", 0.68, "Preparing segmentation output for instance extraction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        colors=colors,
        labels=full_labels,
        probabilities=full_probabilities,
        class_names=np.asarray(CLASS_NAMES),
        voxel_count=np.asarray([len(coordinates)], dtype=np.int64),
        device=np.asarray([device.type]),
    )
    progress("export", 0.70, "Semantic output ready")


if __name__ == "__main__":
    main()
