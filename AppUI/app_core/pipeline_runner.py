from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from MinkUNET.minkunet.data.pcd import read_pcd, write_colored_pcd
from func_build_graph import build_scene_graph
from func_build_temporal_graph import build_temporal_graph
from func_query_llm_temporal import query_llm_temporal
from func_query_local_llm import query_local_llm
from func_segment_minkunet import MinkUNETObjectSegmenter, segment_geometry_minkunet

from .timestamp_utils import parse_pcd_timestamp


APP_RESULTS_ROOT = Path("AppUI/app_results")


class AppRunError(RuntimeError):
    pass


def ensure_results_root(root: Path = APP_RESULTS_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_run_dir(mode: str, run_name: str = "") -> Path:
    root = ensure_results_root()
    mode_dir = root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = run_name.strip().replace(" ", "_") if run_name.strip() else mode
    run_dir = mode_dir / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _notify(
    callback: Optional[Callable[[Dict], None]],
    stage: str,
    message: str,
    progress: float,
    extra: Optional[Dict] = None,
) -> None:
    if callback is None:
        return
    event = {"stage": stage, "message": message, "progress": float(progress)}
    if extra:
        event.update(extra)
    callback(event)


def _safe_load_json(path: Path) -> Dict:
    if not path.exists():
        return {}


def _llm_artifacts(output_dir: Path, json_filename: str, raw_filename: str) -> Dict[str, Optional[str]]:
    llm_dir = output_dir / "LLM"
    json_path = llm_dir / json_filename
    raw_path = llm_dir / raw_filename
    return {
        "llm_response": str(json_path.resolve()) if json_path.exists() else None,
        "llm_raw": str(raw_path.resolve()) if raw_path.exists() else None,
    }
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def run_single_pipeline(
    pcd_path: str,
    run_dir: Path,
    minkunet_config: str,
    checkpoint_path: str,
    temporal_window: int,
    graph_dist_threshold: float,
    run_llm: bool,
    llm_model: str,
    class_dbscan_eps: Optional[Dict[str, float]] = None,
    class_min_points: Optional[Dict[str, int]] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    source = Path(pcd_path)
    if not source.exists():
        raise AppRunError(f"Input PCD does not exist: {source}")

    frame_dir = run_dir / source.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "segmentation", "Running MinkUNET segmentation", 0.1)
    segmented_objects_file = segment_geometry_minkunet(
        pcd_path=str(source),
        output_dir=str(frame_dir),
        config_path=minkunet_config,
        checkpoint_path=checkpoint_path,
        temporal_window=int(max(1, temporal_window)),
        visualize=False,
        class_dbscan_eps=class_dbscan_eps,
        class_min_points=class_min_points,
    )
    if not segmented_objects_file:
        raise AppRunError("Segmentation failed.")

    _notify(progress_callback, "graph", "Building scene graph", 0.45)
    prompt_file = build_scene_graph(
        objects_file=segmented_objects_file,
        output_dir=str(frame_dir),
        dist_threshold=float(graph_dist_threshold),
    )
    if not prompt_file:
        raise AppRunError("Scene graph construction failed.")

    llm_response = None
    if run_llm:
        _notify(progress_callback, "llm", "Running LLM reasoning", 0.75)
        llm_response = query_local_llm(
            prompt_file=prompt_file,
            output_dir=str(frame_dir),
            model_name=llm_model,
        )

    _notify(progress_callback, "finalize", "Collecting outputs", 0.95)
    summary = _safe_load_json(frame_dir / "semantic_summary.json")
    scene_graph = _safe_load_json(frame_dir / "scene_graph.json")

    llm_files = _llm_artifacts(frame_dir, "llm_response_real.json", "llm_response_real.raw.txt")
    llm_status = "skipped"
    if run_llm:
        llm_status = "ok" if llm_files["llm_response"] else "failed"

    run_summary = {
        "mode": "single",
        "run_dir": str(run_dir.resolve()),
        "frame_dir": str(frame_dir.resolve()),
        "source_pcd": str(source.resolve()),
        "outputs": {
            "scene_objects": str((frame_dir / "scene_objects_segmented.json").resolve()),
            "semantic_pcd": str((frame_dir / "semantic_segmentation.pcd").resolve()),
            "semantic_npz": str((frame_dir / "semantic_prediction.npz").resolve()),
            "scene_graph": str((frame_dir / "scene_graph.json").resolve()),
            "prompt": str((frame_dir / "llm_prompt.txt").resolve()),
            "llm_response": llm_files["llm_response"],
            "llm_raw": llm_files["llm_raw"],
        },
        "metrics": {
            "point_count": int(summary.get("point_count", 0)),
            "object_count": int(summary.get("object_count", 0)),
            "class_counts": summary.get("class_counts", {}),
            "graph_nodes": len(scene_graph.get("nodes", [])),
            "graph_edges": len(scene_graph.get("edges", [])),
            "llm_status": llm_status,
        },
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    _notify(progress_callback, "done", "Single-frame pipeline completed", 1.0, {"run_summary": run_summary})
    return run_summary


def run_temporal_pipeline(
    selected_files: List[Path],
    run_dir: Path,
    minkunet_config: str,
    checkpoint_path: str,
    temporal_window: int,
    graph_dist_threshold: float,
    max_track_distance: float,
    movement_threshold: float,
    run_llm: bool,
    llm_model: str,
    class_dbscan_eps: Optional[Dict[str, float]] = None,
    class_min_points: Optional[Dict[str, int]] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    if not selected_files:
        raise AppRunError("No frames selected for temporal run.")

    frame_root = run_dir / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, "init", "Initializing MinkUNET segmenter", 0.02)
    segmenter = MinkUNETObjectSegmenter(
        config_path=minkunet_config,
        checkpoint_path=checkpoint_path,
        temporal_window=int(max(1, temporal_window)),
        class_dbscan_eps=class_dbscan_eps,
        class_min_points=class_min_points,
    )
    class_names = list(segmenter.class_names)

    frame_json_files: List[str] = []
    frame_metrics: List[Dict] = []
    total_class_counts = {name: 0 for name in class_names}

    for frame_idx, pcd_path in enumerate(selected_files):
        t0 = time.perf_counter()
        points, colors = read_pcd(pcd_path)
        pred = segmenter.infer_labels(points=points, colors=colors)
        labels = pred["labels"]
        pred_colors = pred["pred_colors"]

        ts = parse_pcd_timestamp(pcd_path)
        timestamp = ts.isoformat() if ts else pcd_path.stem
        objects = segmenter.objects_from_labels(
            points=points,
            labels=labels,
            frame_id=frame_idx,
            timestamp=timestamp,
        )

        frame_dir = frame_root / pcd_path.stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        objects_json = frame_dir / "scene_objects_segmented.json"
        pred_pcd = frame_dir / "semantic_segmentation.pcd"
        pred_npz = frame_dir / "semantic_prediction.npz"
        frame_summary = frame_dir / "summary.json"

        with objects_json.open("w", encoding="utf-8") as handle:
            json.dump(objects, handle, indent=2)
        write_colored_pcd(pred_pcd, points, pred_colors)
        np.savez_compressed(
            pred_npz,
            points=points.astype(np.float32),
            colors=colors.astype(np.float32),
            labels=labels.astype(np.int64),
            class_names=np.array(class_names),
        )

        class_counts = {name: int((labels == i).sum()) for i, name in enumerate(class_names)}
        for cls_name, cls_count in class_counts.items():
            total_class_counts[cls_name] += cls_count

        infer_sec = time.perf_counter() - t0
        entry = {
            "frame_index": frame_idx,
            "frame_name": pcd_path.stem,
            "source_file": str(pcd_path.resolve()),
            "timestamp": timestamp,
            "num_points": int(points.shape[0]),
            "num_objects": int(len(objects)),
            "inference_seconds": float(infer_sec),
            "class_counts": class_counts,
        }
        frame_metrics.append(entry)

        with frame_summary.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle, indent=2)

        frame_json_files.append(str(objects_json.resolve()))

        # 0.02 -> 0.72 reserved for segmentation loop
        base = 0.02 + 0.70 * float(frame_idx + 1) / float(len(selected_files))
        _notify(
            progress_callback,
            "segmentation",
            f"Segmented frame {frame_idx + 1}/{len(selected_files)}: {pcd_path.name}",
            base,
            {"preview_npz": str(pred_npz.resolve()), "frame_name": pcd_path.stem},
        )

    metrics_df = pd.DataFrame(frame_metrics)
    metrics_df.to_csv(run_dir / "frame_metrics.csv", index=False)
    with (run_dir / "frame_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(frame_metrics, handle, indent=2)

    _notify(progress_callback, "graph", "Building temporal scene graph", 0.80)
    temporal_prompt_file = build_temporal_graph(
        frame_files=frame_json_files,
        output_dir=str(run_dir),
        dist_threshold=float(graph_dist_threshold),
        max_track_distance=float(max_track_distance),
        movement_threshold=float(movement_threshold),
    )
    if not temporal_prompt_file:
        raise AppRunError("Temporal graph construction failed.")

    llm_response = None
    if run_llm:
        _notify(progress_callback, "llm", "Running temporal LLM reasoning", 0.90)
        llm_response = query_llm_temporal(
            prompt_file=temporal_prompt_file,
            output_dir=str(run_dir),
            model_name=llm_model,
        )

    temporal_graph = _safe_load_json(run_dir / "temporal_scene_graph.json")
    llm_files = _llm_artifacts(run_dir, "llm_temporal_response.json", "llm_raw_output.txt")
    llm_status = "skipped"
    if run_llm:
        llm_status = "ok" if llm_files["llm_response"] else "failed"

    run_summary = {
        "mode": "temporal",
        "run_dir": str(run_dir.resolve()),
        "num_frames": len(selected_files),
        "outputs": {
            "frames_dir": str(frame_root.resolve()),
            "frame_metrics_csv": str((run_dir / "frame_metrics.csv").resolve()),
            "temporal_graph": str((run_dir / "temporal_scene_graph.json").resolve()),
            "prompt": str((run_dir / "llm_temporal_prompt.txt").resolve()),
            "llm_response": llm_files["llm_response"],
            "llm_raw": llm_files["llm_raw"],
        },
        "metrics": {
            "total_points": int(metrics_df["num_points"].sum()) if not metrics_df.empty else 0,
            "total_objects": int(metrics_df["num_objects"].sum()) if not metrics_df.empty else 0,
            "avg_inference_seconds": float(metrics_df["inference_seconds"].mean()) if not metrics_df.empty else 0.0,
            "fps_estimate": float(1.0 / metrics_df["inference_seconds"].mean()) if not metrics_df.empty and metrics_df["inference_seconds"].mean() > 1e-9 else 0.0,
            "class_counts": total_class_counts,
            "track_summary": temporal_graph.get("summary", {}),
            "llm_status": llm_status,
        },
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    _notify(progress_callback, "done", "Temporal pipeline completed", 1.0, {"run_summary": run_summary})
    return run_summary
