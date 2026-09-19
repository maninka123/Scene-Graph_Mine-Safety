from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from mine_safety.config import load_config
from mine_safety.reasoning.llm import LocalQwenReasoner
from mine_safety.schemas import SafetyAlert, SceneGraph

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
_reasoner: LocalQwenReasoner | None = None
_model_path: Path | None = None
_load_ms: float | None = None
_lock = threading.Lock()


def _snapshot_from_cache(cache_root: Path) -> Path | None:
    snapshots = cache_root / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_model_path() -> Path | None:
    configured = os.getenv("MINEGRAPH_QWEN_MODEL")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    candidates = [
        ROOT / "models" / "Qwen2.5-3B-Instruct",
        ROOT.parent / "LLM" / "Model_Cache",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    if candidates[0].is_dir():
        return candidates[0]
    for cache in candidates[1:]:
        snapshot = _snapshot_from_cache(cache)
        if snapshot:
            return snapshot
    return None


def model_status() -> dict[str, Any]:
    path = find_model_path()
    cuda_available = False
    gpu_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        pass
    return {
        "model_id": MODEL_ID,
        "available": path is not None,
        "loaded": _reasoner is not None,
        "local_path": str(path) if path else None,
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "load_ms": _load_ms,
        "prompt_profile": "Appendix A contextual safety reasoning",
        "validation": "Strict JSON schema + exact object-ID grounding",
    }


def _get_reasoner() -> LocalQwenReasoner:
    global _load_ms, _model_path, _reasoner
    if _reasoner is not None:
        return _reasoner
    path = find_model_path()
    if path is None:
        raise RuntimeError(
            "Qwen2.5-3B-Instruct was not found locally. Set MINEGRAPH_QWEN_MODEL "
            "to a downloaded model directory."
        )
    config = load_config(ROOT / "configs" / "paper.yaml")["reasoning"].copy()
    config["contextual_model"] = str(path)
    started = time.perf_counter()
    reasoner = LocalQwenReasoner(config)
    # Load now so model loading and generation timings remain distinct.
    reasoner.load()
    _load_ms = (time.perf_counter() - started) * 1000
    _model_path = path
    _reasoner = reasoner
    return reasoner


def reason_scene(graph_payload: dict[str, Any], alert_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    graph = SceneGraph.model_validate(graph_payload)
    alerts = [SafetyAlert.model_validate(alert) for alert in alert_payloads]
    temporal = {
        "window_seconds": 10.0,
        "frames": [graph.model_dump(mode="json")],
        "tracks": {},
    }
    with _lock:
        was_loaded = _reasoner is not None
        reasoner = _get_reasoner()
        generation_started = time.perf_counter()
        result = reasoner.reason(graph, temporal, alerts)
        generation_ms = (time.perf_counter() - generation_started) * 1000
    status = model_status()
    return {
        "assessment": result.model_dump(mode="json"),
        "model": {
            **status,
            "loaded_for_request": not was_loaded,
            "model_path_name": _model_path.name if _model_path else None,
        },
        "timing": {
            "model_load_ms": _load_ms if not was_loaded else 0.0,
            "generation_ms": generation_ms,
            "total_ms": generation_ms + ((_load_ms or 0.0) if not was_loaded else 0.0),
        },
        "grounding": {
            "input_nodes": len(graph.nodes),
            "input_edges": len(graph.edges),
            "deterministic_alerts_supplied": len(alerts),
            "output_schema_valid": True,
            "object_ids_validated": True,
        },
    }
