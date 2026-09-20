from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.rules import SafetyRuleEngine
from mine_safety.schemas import SceneNode

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
DEMO_ID = "PC_20260122_153641662"
CLASS_COLOURS = {
    "wall": "#8E8E93",
    "equipment": "#FF9F0A",
    "human": "#FF453A",
    "conveyor": "#30B0C7",
    "roof": "#AF52DE",
    "other": "#64D2FF",
    "unlabeled": "#D1D1D6",
    "personnel": "#FF453A",
}


def _hex_to_rgb(value: str) -> list[float]:
    value = value.lstrip("#")
    return [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _normalise_node(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw["label"])
    return {
        "id": f"{label}-{int(raw['id']):02d}",
        "label": label,
        "centroid": [round(float(value), 5) for value in raw["centroid"]],
        "bbox_dimensions": [round(float(value), 5) for value in raw["dimensions"]],
        "orientation": [1.0, 0.0, 0.0],
        "volume_m3": round(float(raw["volume"]), 5),
        "voxel_count": int(raw["point_count"]),
        "confidence": float(raw["confidence"]),
        "entropy": 0.0,
        "is_anomaly": False,
        "active": True,
        "velocity": [0.0, 0.0, 0.0],
        "movement_state": "unknown",
        "synthetic": False,
    }


def _reference_timings() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads((DEMO_DIR / "reference_timing_rtx4080.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    names = {
        "Voxelisation and sparse tensor construction": ("voxelise", "Voxelise + sparse tensor"),
        "Sparse 3D perception (forward pass)": ("minkunet", "MinkUNet inference"),
        "Uncertainty map computation and anomaly clustering": ("uncertainty", "Uncertainty + anomalies"),
        "Semantic object extraction (required pre-graph)": ("instances", "Instance extraction"),
        "Scene graph construction": ("graph", "Scene graph"),
        "Temporal graph update and kinematic extraction": ("temporal", "Temporal update"),
        "Deterministic rule evaluation": ("rules", "Safety rules"),
        "LLM full generation (contextual reasoning)": ("qwen", "Qwen reasoning"),
    }
    timings = []
    for row in rows:
        if row["stage"] not in names:
            continue
        stage_id, label = names[row["stage"]]
        timings.append({
            "id": stage_id,
            "label": label,
            "mean_ms": row["mean_ms"],
            "p90_ms": row["p90_ms"],
            "status": row["status"],
            "source": "recorded",
            "notes": row["notes"],
        })
    environment = payload["metadata"]["environment"]
    return timings, {
        "gpu": environment.get("gpu"),
        "cuda_available": environment.get("cuda_available"),
        "generated_at": environment.get("generated_at"),
        "repeats": payload["metadata"]["args"].get("frame_repeats"),
        "model": payload["metadata"]["args"].get("llm_model"),
    }


def analyse_nodes(nodes: list[dict[str, Any]], edge_distance_m: float) -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "paper.yaml")
    parsed = []
    for node in nodes:
        clean = {key: value for key, value in node.items() if key in SceneNode.model_fields}
        parsed.append(SceneNode.model_validate(clean))
    graph = build_scene_graph(
        parsed,
        graph_id=f"{DEMO_ID}-interactive",
        edge_distance_m=edge_distance_m,
        mean_intensity=90.0,
        metadata={
            "source": f"data/demo/{DEMO_ID}.pcd",
            "edge_distance_m": edge_distance_m,
            "interactive": True,
        },
    )
    alerts = SafetyRuleEngine(config["rules"]).evaluate(graph)
    return {
        "graph": graph.model_dump(mode="json"),
        "alerts": [alert.model_dump(mode="json") for alert in alerts],
    }


@lru_cache(maxsize=1)
def load_demo(max_points: int = 12000) -> dict[str, Any]:
    prediction = np.load(DEMO_DIR / "semantic_prediction.npz")
    points = prediction["points"]
    rgb = prediction["colors"]
    labels = prediction["labels"]
    class_names = [str(name) for name in prediction["class_names"].tolist()]
    indices = np.linspace(0, len(points) - 1, min(max_points, len(points)), dtype=np.int64)

    sampled_labels = labels[indices]
    semantic_colours = []
    label_names = []
    for class_index in sampled_labels:
        name = class_names[int(class_index)] if 0 <= int(class_index) < len(class_names) else "unlabeled"
        label_names.append(name)
        semantic_colours.append(_hex_to_rgb(CLASS_COLOURS.get(name, CLASS_COLOURS["other"])))

    raw_nodes = json.loads((DEMO_DIR / "scene_objects_segmented.json").read_text(encoding="utf-8"))
    nodes = [_normalise_node(node) for node in raw_nodes]
    analysis = analyse_nodes(nodes, edge_distance_m=8.0)
    timings, timing_environment = _reference_timings()
    class_counts = {
        name: int(np.count_nonzero(labels == index)) for index, name in enumerate(class_names)
    }
    class_counts["unlabeled"] = int(np.count_nonzero(labels < 0))

    return {
        "demo": {
            "id": DEMO_ID,
            "filename": f"{DEMO_ID}.pcd",
            "source": "Simulator training dataset",
            "point_count": len(points),
            "displayed_points": len(indices),
            "voxel_count": 11877,
            "object_count": len(nodes),
            "recorded_edge_count": 17,
            "mean_intensity": round(float(rgb.mean() * 255.0), 2),
            "classes": class_names,
            "class_counts": class_counts,
        },
        "point_cloud": {
            "positions": np.round(points[indices], 5).tolist(),
            "rgb": np.round(rgb[indices], 4).tolist(),
            "semantic": semantic_colours,
            "labels": label_names,
        },
        "nodes": nodes,
        "graph": analysis["graph"],
        "alerts": analysis["alerts"],
        "timings": timings,
        "timing_environment": timing_environment,
        "class_colours": CLASS_COLOURS,
        "pipeline": [
            {"id": "pcd", "label": "PCD input", "detail": "48,000 XYZRGB points", "kind": "live"},
            {"id": "voxelise", "label": "Voxelise", "detail": "1 cm sparse voxels", "kind": "recorded"},
            {"id": "minkunet", "label": "MinkUNet", "detail": "Sparse 3D semantic labels", "kind": "recorded"},
            {"id": "instances", "label": "Detections", "detail": "Class-aware DBSCAN", "kind": "recorded"},
            {"id": "graph", "label": "Scene graph", "detail": "Live Python rebuild", "kind": "live"},
            {"id": "rules", "label": "Safety rules", "detail": "Live deterministic checks", "kind": "live"},
            {"id": "qwen", "label": "Qwen", "detail": "Optional contextual reasoning", "kind": "optional"},
        ],
    }


def run_interactive(nodes: list[dict[str, Any]], edge_distance_m: float) -> dict[str, Any]:
    started = time.perf_counter()
    graph_started = time.perf_counter()
    result = analyse_nodes(nodes, edge_distance_m)
    graph_ms = (time.perf_counter() - graph_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    result["live_timings"] = {
        "scene_graph_ms": round(graph_ms, 4),
        "total_postprocess_ms": round(total_ms, 4),
    }
    return result
