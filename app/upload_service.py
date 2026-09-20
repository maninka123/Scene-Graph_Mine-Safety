from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.demo_service import CLASS_COLOURS, ROOT, _hex_to_rgb
from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.perception.instances import extract_instances
from mine_safety.rules import SafetyRuleEngine

RUNTIME_DIR = ROOT / ".runtime" / "uploads"
CHECKPOINT = ROOT / "checkpoints" / "demo_adapter" / "semantic_best.pt"
WORKER = ROOT / "scripts" / "segment_upload_wsl.py"
ALLOWED_SUFFIXES = {".pcd", ".ply", ".npz"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
DISPLAY_POINTS = 12_000

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def perception_status() -> dict[str, Any]:
    if os.name == "nt":
        runtime_ready = bool(
            shutil.which("wsl")
            and (ROOT.parent / ".venv_wsl" / "bin" / "activate").is_file()
        )
        runtime = "WSL 2 + CUDA MinkowskiEngine"
    else:
        try:
            import MinkowskiEngine  # noqa: F401

            runtime_ready = True
        except ImportError:
            runtime_ready = False
        runtime = "Native CUDA MinkowskiEngine"
    return {
        "available": bool(runtime_ready and CHECKPOINT.is_file() and WORKER.is_file()),
        "runtime": runtime,
        "checkpoint": "demo_adapter/semantic_best.pt",
        "accepted_formats": sorted(ALLOWED_SUFFIXES),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


def create_job(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Choose a .pcd, .ply, or .npz point cloud")
    if not content:
        raise ValueError("The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Point cloud exceeds the 250 MB upload limit")
    if not perception_status()["available"]:
        raise RuntimeError("The local WSL CUDA segmentation runtime is not available")

    job_id = uuid.uuid4().hex
    safe_name = f"input{suffix}"
    job_dir = RUNTIME_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / safe_name
    input_path.write_bytes(content)
    job = {
        "id": job_id,
        "filename": Path(filename).name,
        "bytes": len(content),
        "status": "queued",
        "stage": "upload",
        "progress": 0.04,
        "message": "Upload received and stored locally",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    with _lock:
        _jobs[job_id] = job
    threading.Thread(
        target=_run_job,
        args=(job_id, input_path, job_dir / "segmentation.npz"),
        daemon=True,
    ).start()
    return public_job(job_id)


def public_job(job_id: str) -> dict[str, Any]:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(job_id)
        return dict(_jobs[job_id])


def _update(job_id: str, **values: Any) -> None:
    with _lock:
        _jobs[job_id].update(values)


def _wsl_path(path: Path) -> str:
    result = subprocess.run(
        ["wsl", "wslpath", "-a", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_worker(job_id: str, input_path: Path, output_path: Path) -> None:
    if os.name != "nt":
        process = subprocess.Popen(
            [
                sys.executable,
                str(WORKER),
                "--input",
                str(input_path),
                "--checkpoint",
                str(CHECKPOINT),
                "--output",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _consume_worker(job_id, process)
        return
    linux_root = _wsl_path(ROOT)
    linux_input = _wsl_path(input_path)
    linux_output = _wsl_path(output_path)
    linux_checkpoint = _wsl_path(CHECKPOINT)
    linux_venv = _wsl_path(ROOT.parent / ".venv_wsl" / "bin" / "activate")
    command = (
        f"cd {shlex.quote(linux_root)} && source {shlex.quote(linux_venv)} && "
        "OMP_NUM_THREADS=12 PYTHONPATH=src python scripts/segment_upload_wsl.py "
        f"--input {shlex.quote(linux_input)} --checkpoint {shlex.quote(linux_checkpoint)} "
        f"--output {shlex.quote(linux_output)}"
    )
    process = subprocess.Popen(
        ["wsl", "-e", "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _consume_worker(job_id, process)


def _consume_worker(job_id: str, process: subprocess.Popen[str]) -> None:
    log_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.strip()
        if clean.startswith("MINEGRAPH_PROGRESS "):
            update = json.loads(clean.removeprefix("MINEGRAPH_PROGRESS "))
            _update(job_id, status="running", **update)
        elif clean:
            log_lines.append(clean)
    return_code = process.wait()
    if return_code:
        detail = log_lines[-1] if log_lines else f"WSL worker exited with code {return_code}"
        raise RuntimeError(detail)


def _build_result(
    job_id: str, filename: str, output_path: Path, timings: dict[str, float]
) -> dict[str, Any]:
    data = np.load(output_path)
    points = np.asarray(data["points"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    probabilities = np.asarray(data["probabilities"], dtype=np.float32)
    class_names = [str(name) for name in data["class_names"].tolist()]
    voxel_count = int(data["voxel_count"][0])

    valid = labels >= 0
    config = load_config(ROOT / "configs" / "paper.yaml")
    voxel_keys = np.floor(points / config["perception"]["voxel_size_m"]).astype(np.int32)
    _, representatives = np.unique(voxel_keys, axis=0, return_index=True)
    representatives = representatives[valid[representatives]]
    _update(
        job_id,
        stage="instances",
        progress=0.76,
        message="Clustering semantic points into object instances",
    )
    instance_started = time.perf_counter()
    nodes = extract_instances(
        points[representatives],
        labels[representatives],
        probabilities[representatives],
        class_names,
        epsilon_m=0.25,
        minimum_voxels=20,
    )
    timings["instances_ms"] = (time.perf_counter() - instance_started) * 1000

    _update(
        job_id,
        stage="graph",
        progress=0.86,
        message="Constructing the object-relation scene graph",
    )
    graph_started = time.perf_counter()
    graph = build_scene_graph(
        nodes,
        graph_id=f"upload-{Path(filename).stem}",
        edge_distance_m=config["graph"]["edge_distance_m"],
        mean_intensity=float(colors.mean() * 255.0),
        metadata={
            "source": filename,
            "uploaded": True,
            "segmentation_checkpoint": "demo_adapter/semantic_best.pt",
        },
    )
    timings["graph_ms"] = (time.perf_counter() - graph_started) * 1000
    _update(
        job_id,
        stage="rules",
        progress=0.94,
        message="Evaluating deterministic safety rules",
    )
    rule_started = time.perf_counter()
    alerts = SafetyRuleEngine(config["rules"]).evaluate(graph)
    timings["rules_ms"] = (time.perf_counter() - rule_started) * 1000

    indices = np.linspace(0, len(points) - 1, min(DISPLAY_POINTS, len(points)), dtype=np.int64)
    sampled_labels = labels[indices]
    label_names = [
        class_names[index] if 0 <= index < len(class_names) else "unlabeled"
        for index in sampled_labels
    ]
    semantic = [
        _hex_to_rgb(CLASS_COLOURS.get(name, CLASS_COLOURS["other"])) for name in label_names
    ]
    counts = {name: int(np.count_nonzero(labels == index)) for index, name in enumerate(class_names)}
    counts["unlabeled"] = int(np.count_nonzero(labels < 0))
    node_payloads = [{**node.model_dump(mode="json"), "synthetic": False} for node in nodes]

    return {
        "demo": {
            "id": f"upload-{Path(filename).stem}",
            "filename": filename,
            "source": "User upload · processed locally",
            "point_count": len(points),
            "displayed_points": len(indices),
            "voxel_count": voxel_count,
            "object_count": len(nodes),
            "recorded_edge_count": len(graph.edges) // 2,
            "mean_intensity": round(float(colors.mean() * 255.0), 2),
            "classes": class_names,
            "class_counts": counts,
            "uploaded": True,
            "model_provenance": "Local trained MinkUNet checkpoint",
        },
        "point_cloud": {
            "positions": np.round(points[indices], 5).tolist(),
            "rgb": np.round(colors[indices], 4).tolist(),
            "semantic": semantic,
            "labels": label_names,
        },
        "nodes": node_payloads,
        "graph": graph.model_dump(mode="json"),
        "alerts": [alert.model_dump(mode="json") for alert in alerts],
        "class_colours": CLASS_COLOURS,
        "upload_timings": {key: round(value, 3) for key, value in timings.items()},
        "pipeline": [
            {"id": "pcd", "label": "Point cloud", "detail": f"{len(points):,} XYZRGB points", "kind": "complete"},
            {"id": "voxelise", "label": "Voxelise", "detail": f"{voxel_count:,} sparse voxels", "kind": "complete"},
            {"id": "minkunet", "label": "MinkUNet", "detail": "Local GPU inference", "kind": "complete"},
            {"id": "instances", "label": "Detections", "detail": f"{len(nodes)} clustered instances", "kind": "complete"},
            {"id": "graph", "label": "Scene graph", "detail": f"{len(graph.edges)} directed relations", "kind": "live"},
            {"id": "rules", "label": "Safety rules", "detail": f"{len(alerts)} grounded findings", "kind": "live"},
            {"id": "qwen", "label": "Qwen", "detail": "Ready for contextual reasoning", "kind": "optional"},
        ],
    }


def _run_job(job_id: str, input_path: Path, output_path: Path) -> None:
    started = time.perf_counter()
    try:
        _update(job_id, status="running", stage="upload", progress=0.08, message="Starting the local CUDA inference runtime")
        inference_started = time.perf_counter()
        _run_worker(job_id, input_path, output_path)
        timings = {"segmentation_ms": (time.perf_counter() - inference_started) * 1000}
        with _lock:
            filename = _jobs[job_id]["filename"]
        result = _build_result(job_id, filename, output_path, timings)
        result["upload_timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        _update(
            job_id,
            status="complete",
            stage="complete",
            progress=1.0,
            message="Point cloud pipeline complete",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - background boundary must expose worker failures
        _update(
            job_id,
            status="failed",
            stage="error",
            message="The point cloud could not be processed",
            error=str(exc),
        )
