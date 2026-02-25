from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch


def _model_id_from_dir(model_dir: Path) -> str:
    # Example: models--Qwen--Qwen2.5-7B-Instruct -> Qwen/Qwen2.5-7B-Instruct
    name = model_dir.name
    if name.startswith("models--"):
        name = name[len("models--") :]
    return name.replace("--", "/")


def _latest_snapshot_dir(model_dir: Path) -> Optional[Path]:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _weights_size_bytes(snapshot_dir: Path) -> int:
    total = 0
    for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        for p in snapshot_dir.glob(ext):
            if p.is_file():
                total += p.stat().st_size
    return total


def _estimate_required_vram_gb(weights_gb: float) -> Optional[float]:
    if weights_gb <= 0:
        return None
    # Rule-of-thumb for inference with kv-cache/headroom.
    return round(weights_gb * 1.2 + 2.0, 1)


def discover_local_models(workspace_root: Path) -> List[Dict]:
    roots = [
        workspace_root / "LLM" / "Model_Cache",
        workspace_root / "LLM" / "DeepSeek_Model",
    ]
    model_dirs: List[Path] = []
    for root in roots:
        if root.exists():
            model_dirs.extend([p for p in root.glob("models--*") if p.is_dir()])

    seen = set()
    rows: List[Dict] = []
    for model_dir in model_dirs:
        model_id = _model_id_from_dir(model_dir)
        if model_id in seen:
            continue
        seen.add(model_id)

        snapshot_dir = _latest_snapshot_dir(model_dir)
        weights_gb = 0.0
        required_vram_gb = None
        snapshot_path = None
        if snapshot_dir is not None:
            snapshot_path = str(snapshot_dir)
            weights_bytes = _weights_size_bytes(snapshot_dir)
            weights_gb = round(weights_bytes / (1024**3), 2)
            required_vram_gb = _estimate_required_vram_gb(weights_gb)

        rows.append(
            {
                "model_id": model_id,
                "source_dir": str(model_dir),
                "snapshot_dir": snapshot_path,
                "weights_gb": weights_gb,
                "required_vram_gb": required_vram_gb,
            }
        )
    rows.sort(key=lambda x: x["model_id"].lower())
    return rows


def get_device_report() -> Dict:
    report = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "device_text": "CPU only",
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        vram = round(prop.total_memory / (1024**3), 1)
        report["gpu_name"] = prop.name
        report["gpu_vram_gb"] = vram
        report["device_text"] = f"CUDA GPU: {prop.name} ({vram} GB VRAM)"
    return report


def compatibility_message(model_row: Optional[Dict], device_report: Dict) -> Dict[str, str]:
    if model_row is None:
        return {
            "status": "info",
            "title": "Custom/Unknown model",
            "message": "Model size not found locally. Compatibility estimate is unavailable.",
            "icon": "ℹ️",
        }

    req = model_row.get("required_vram_gb")
    weights = model_row.get("weights_gb", 0.0)
    model_id = model_row.get("model_id", "Model")

    if not device_report.get("cuda_available"):
        if weights > 0 and weights <= 4.0:
            return {
                "status": "warning",
                "title": f"{model_id}: CPU possible (slow)",
                "message": "This can run on CPU, but latency will be high. GPU is recommended.",
                "icon": "⚠️",
            }
        return {
            "status": "error",
            "title": f"{model_id}: Not ideal on CPU",
            "message": "Likely very slow or memory-heavy on CPU. Prefer a smaller model or use a CUDA GPU.",
            "icon": "⛔",
        }

    gpu_vram = device_report.get("gpu_vram_gb")
    if req is None or gpu_vram is None:
        return {
            "status": "info",
            "title": f"{model_id}: Compatibility unknown",
            "message": "Could not estimate VRAM requirement. Try a quick run to validate.",
            "icon": "ℹ️",
        }

    if gpu_vram >= req:
        return {
            "status": "success",
            "title": f"{model_id}: Can run on this GPU",
            "message": f"Estimated requirement ~{req} GB VRAM, available {gpu_vram} GB.",
            "icon": "✅",
        }

    if gpu_vram >= max(req - 3.0, req * 0.75):
        return {
            "status": "warning",
            "title": f"{model_id}: Can run, but not ideal",
            "message": (
                f"Estimated requirement ~{req} GB VRAM, available {gpu_vram} GB. "
                "Use smaller context, offloading, or a smaller model for stability."
            ),
            "icon": "⚠️",
        }

    return {
        "status": "error",
        "title": f"{model_id}: Likely cannot run well on this GPU",
        "message": (
            f"Estimated requirement ~{req} GB VRAM, available {gpu_vram} GB. "
            "Use a smaller model or larger GPU."
        ),
        "icon": "⛔",
    }
