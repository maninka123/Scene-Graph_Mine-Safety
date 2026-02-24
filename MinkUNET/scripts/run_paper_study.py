from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from _common import REPO_ROOT, abs_path, load_cfg
from minkunet.data.pcd import read_pcd
from minkunet.data.preprocess import preprocess_sample
from minkunet.engine.infer import TemporalMajoritySmoother, infer_single_frame
from minkunet.models.network import SemanticMinkUNet, sparse_tensor_from_batch
from minkunet.utils.io import ensure_dir, read_jsonl, resolve_existing_dir, write_json, write_jsonl
from minkunet.utils.metrics import compute_segmentation_metrics, confusion_matrix


def _deep_update(base: Dict, override: Dict) -> Dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_cmd(args: List[str], cwd: Path) -> None:
    print(f"[Run] {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), check=True)


def _load_model(cfg: Dict, checkpoint_path: Path, device: torch.device) -> SemanticMinkUNet:
    model = SemanticMinkUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        feature_dim=int(cfg["model"]["feature_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        width=int(cfg["model"]["width"]),
    ).to(device)
    state = torch.load(str(checkpoint_path), map_location=str(device))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _collect_pcds(input_path: Path, limit: int) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pcd":
        return [input_path]
    files = sorted(input_path.glob("*.pcd"))
    return files[:limit] if limit > 0 else files


def _hist_from_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    valid = labels[labels >= 0]
    if valid.size == 0:
        return np.zeros(num_classes, dtype=np.float64)
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    denom = max(counts.sum(), 1.0)
    return counts / denom


def benchmark_inference_runtime(
    cfg: Dict,
    checkpoint_path: Path,
    input_path: Path,
    frames: int,
    temporal_window: int,
) -> Dict:
    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = _load_model(cfg=cfg, checkpoint_path=checkpoint_path, device=device)
    pcd_files = _collect_pcds(input_path, frames)
    if not pcd_files:
        raise RuntimeError(f"No .pcd files found for inference benchmark: {input_path}")

    smoother = TemporalMajoritySmoother(window=int(max(1, temporal_window))) if temporal_window > 1 else None

    latencies = []
    hist_series = []
    max_mem = 0.0
    for pcd in pcd_files:
        points, colors = read_pcd(pcd)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = infer_single_frame(model=model, points=points, colors=colors, cfg=cfg, device=device, temporal_smoother=smoother)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            max_mem = max(max_mem, float(torch.cuda.max_memory_allocated(device) / (1024.0**2)))
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        hist_series.append(_hist_from_labels(out["point_predictions_full"], num_classes=int(cfg["model"]["num_classes"])))

    avg_latency = float(np.mean(latencies))
    fps = float(1.0 / avg_latency) if avg_latency > 0 else 0.0
    stability = 1.0
    if len(hist_series) > 1:
        deltas = []
        for i in range(len(hist_series) - 1):
            deltas.append(float(np.abs(hist_series[i + 1] - hist_series[i]).sum() * 0.5))
        stability = float(1.0 - np.mean(deltas))

    return {
        "frames": int(len(pcd_files)),
        "temporal_window": int(temporal_window),
        "avg_latency_sec": avg_latency,
        "fps": fps,
        "gpu_memory_mb_peak": max_mem,
        "stability_score": stability,
    }


def _apply_corruption(
    points: np.ndarray,
    colors: np.ndarray,
    mode: str,
    cfg: Dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mode == "clean":
        return points, colors, None

    pts = points.copy()
    cols = colors.copy()

    robustness_cfg = cfg["paper_study"]["robustness"]
    if mode == "noise":
        std = float(robustness_cfg["noise_std"])
        pts = pts + np.random.normal(0.0, std, size=pts.shape).astype(np.float32)
        return pts, cols, None
    elif mode == "occlusion":
        ratio = float(robustness_cfg["occlusion_ratio"])
        keep = np.random.rand(pts.shape[0]) > ratio
        if keep.sum() < 64:
            keep[np.random.choice(pts.shape[0], size=min(64, pts.shape[0]), replace=False)] = True
        pts = pts[keep]
        cols = cols[keep]
        return pts, cols, keep
    elif mode == "color_degradation":
        factor = float(robustness_cfg["color_degradation"])
        gray = cols.mean(axis=1, keepdims=True).repeat(3, axis=1)
        cols = (1.0 - factor) * cols + factor * gray
        cols = np.clip(cols, 0.0, 1.0)
        return pts, cols, None
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")


@torch.no_grad()
def evaluate_robustness(
    cfg: Dict,
    checkpoint_path: Path,
    manifest_path: Path,
    mode: str,
) -> Dict:
    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = _load_model(cfg=cfg, checkpoint_path=checkpoint_path, device=device)
    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"No records in manifest: {manifest_path}")

    num_classes = int(cfg["model"]["num_classes"])
    class_names = cfg["model"]["class_names"]
    ignore_index = int(cfg["finetune"].get("ignore_index", -1))
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for row in rows:
        npz = np.load(abs_path(row["label_npz"]))
        points = npz["points"].astype(np.float32)
        colors = npz["colors"].astype(np.float32)
        labels = npz["labels"].astype(np.int64)

        points, colors, keep_mask = _apply_corruption(points, colors, mode=mode, cfg=cfg)
        if points.shape[0] == 0:
            continue

        if keep_mask is not None:
            labels = labels[keep_mask]

        processed = preprocess_sample(
            points=points,
            colors=colors,
            labels=labels,
            data_cfg=cfg["data"],
            max_points=int(cfg["data"]["max_points_eval"]),
            augmentation_cfg=None,
        )
        voxel_labels = processed["voxel_labels"]
        if voxel_labels is None or voxel_labels.size == 0:
            continue

        coords = processed["coords"]
        feats = processed["features"]
        batch_col = np.zeros((coords.shape[0], 1), dtype=np.int32)
        coords_batched = np.concatenate([batch_col, coords], axis=1)

        sparse_input = sparse_tensor_from_batch(
            torch.from_numpy(coords_batched).int(),
            torch.from_numpy(feats).float(),
            device=device,
        )
        logits = model(sparse_input).F
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        hist += confusion_matrix(preds, voxel_labels.astype(np.int64), num_classes=num_classes, ignore_index=ignore_index)

    metrics = compute_segmentation_metrics(hist, class_names=class_names)
    return metrics


def _subset_manifest(input_manifest: Path, out_manifest: Path, fraction: float, seed: int) -> int:
    rows = read_jsonl(input_manifest)
    if not rows:
        write_jsonl(out_manifest, [])
        return 0
    rng = random.Random(seed)
    k = max(1, int(len(rows) * fraction))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    selected = [rows[i] for i in sorted(idx[:k])]
    write_jsonl(out_manifest, selected)
    return len(selected)


def run_finetune_and_validate(
    base_cfg: Dict,
    patch: Dict,
    run_id: str,
    run_root: Path,
    train_manifest: Path | None = None,
    val_manifest: Path | None = None,
) -> Dict:
    cfg = _deep_update(base_cfg, patch)
    cfg["logging"] = cfg.get("logging", {})
    cfg["logging"]["run_name"] = run_id
    cfg["checkpoint"] = cfg.get("checkpoint", {})
    ckpt_dir = ensure_dir(run_root / "checkpoints" / run_id)
    cfg["checkpoint"]["dir"] = str(ckpt_dir)
    cfg["finetune"]["checkpoint_name"] = "semantic_best.pt"

    cfg_path = run_root / "configs" / f"{run_id}.yaml"
    _write_yaml(cfg_path, cfg)

    cmd_train = [sys.executable, "MinkUNET/scripts/train_segmentation.py", "--config", str(cfg_path)]
    if train_manifest is not None:
        cmd_train += ["--train-manifest", str(train_manifest)]
    if val_manifest is not None:
        cmd_train += ["--val-manifest", str(val_manifest)]
    _run_cmd(cmd_train, cwd=REPO_ROOT)

    val_out = ensure_dir(run_root / "runs" / run_id / "validation")
    ckpt_path = ckpt_dir / "semantic_best.pt"
    cmd_val = [
        sys.executable,
        "MinkUNET/scripts/validate_segmentation.py",
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(ckpt_path),
        "--output-dir",
        str(val_out),
    ]
    if val_manifest is not None:
        cmd_val += ["--manifest", str(val_manifest)]
    _run_cmd(cmd_val, cwd=REPO_ROOT)

    metrics = json.loads((val_out / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "miou": float(metrics["miou"]),
        "accuracy": float(metrics["accuracy"]),
        "val_loss": float(metrics.get("loss", 0.0)),
    }
    for cname, payload in metrics["per_class"].items():
        result[f"iou_{cname}"] = float(payload["iou"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full paper experiment suite in isolated result folders.")
    parser.add_argument("--config", default="MinkUNET/configs/paper_study.yaml")
    parser.add_argument("--no-visualize", action="store_true", help="Skip final summary plot generation.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    ps = cfg["paper_study"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(ps.get("run_name", "paper_study")).strip() or "paper_study"
    run_root = ensure_dir(abs_path(ps.get("output_root", "Results/Paper_Study")) / f"{run_name}_{stamp}")
    ensure_dir(run_root / "configs")
    ensure_dir(run_root / "summaries")
    ensure_dir(run_root / "runs")

    write_json(run_root / "resolved_paper_study_config.json", cfg)
    print(f"[Study] Output root: {run_root}")

    manifests_dir = abs_path(cfg["data"]["manifests_dir"])
    train_manifest = manifests_dir / "labeled_train.jsonl"
    val_manifest = manifests_dir / "labeled_val.jsonl"
    unlabeled_root = resolve_existing_dir(cfg["data"]["unlabeled_roots"], root=REPO_ROOT)

    train_cfg = ps.get("train", {})
    if train_cfg.get("finetune_epochs") is not None:
        cfg["train"]["epochs"] = int(train_cfg["finetune_epochs"])
    if train_cfg.get("batch_size") is not None:
        cfg["train"]["batch_size"] = int(train_cfg["batch_size"])

    summary_rows = {
        "scratch_vs_pretrained": [],
        "ablation_voxel_size": [],
        "ablation_rgb": [],
        "ablation_labeled_data_size": [],
        "ablation_temporal_smoothing": [],
        "runtime_benchmark": [],
        "robustness": [],
    }
    pretrained_ckpt_path = abs_path(ps["assets"]["pretrained_checkpoint"])
    needs_pretrained = any(
        [
            bool(ps.get("scratch_vs_pretrained", {}).get("enabled", True)),
            bool(ps.get("ablations", {}).get("voxel_size", {}).get("enabled", False)),
            bool(ps.get("ablations", {}).get("rgb_on_off", {}).get("enabled", False)),
            bool(ps.get("ablations", {}).get("labeled_data_size", {}).get("enabled", False)),
        ]
    )
    if needs_pretrained and not pretrained_ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint missing: {pretrained_ckpt_path}. "
            "Run contrastive pretraining first or update paper_study.assets.pretrained_checkpoint."
        )

    # 1) Scratch vs Contrastive-pretrained
    if ps.get("scratch_vs_pretrained", {}).get("enabled", True):
        row_pre = run_finetune_and_validate(
            base_cfg=cfg,
            patch={"finetune": {"pretrained_checkpoint": str(pretrained_ckpt_path)}},
            run_id="scratch_vs_pretrained_pretrained",
            run_root=run_root,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
        )
        row_pre["mode"] = "contrastive_pretrained"
        summary_rows["scratch_vs_pretrained"].append(row_pre)

        row_scratch = run_finetune_and_validate(
            base_cfg=cfg,
            patch={"finetune": {"pretrained_checkpoint": ""}},
            run_id="scratch_vs_pretrained_scratch",
            run_root=run_root,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
        )
        row_scratch["mode"] = "scratch"
        summary_rows["scratch_vs_pretrained"].append(row_scratch)

    # 2) Ablation: voxel size
    vox_cfg = ps.get("ablations", {}).get("voxel_size", {})
    if vox_cfg.get("enabled", False):
        for voxel_size in vox_cfg.get("values", []):
            row = run_finetune_and_validate(
                base_cfg=cfg,
                patch={
                    "data": {"voxel_size": float(voxel_size)},
                    "finetune": {"pretrained_checkpoint": str(pretrained_ckpt_path)},
                },
                run_id=f"ablation_voxel_{str(voxel_size).replace('.', 'p')}",
                run_root=run_root,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
            )
            row["voxel_size"] = float(voxel_size)
            summary_rows["ablation_voxel_size"].append(row)

    # 3) Ablation: RGB on/off
    rgb_cfg = ps.get("ablations", {}).get("rgb_on_off", {})
    if rgb_cfg.get("enabled", False):
        for use_rgb in rgb_cfg.get("values", [True, False]):
            row = run_finetune_and_validate(
                base_cfg=cfg,
                patch={
                    "data": {"use_rgb": bool(use_rgb)},
                    "finetune": {"pretrained_checkpoint": str(pretrained_ckpt_path)},
                },
                run_id=f"ablation_rgb_{'on' if use_rgb else 'off'}",
                run_root=run_root,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
            )
            row["use_rgb"] = bool(use_rgb)
            summary_rows["ablation_rgb"].append(row)

    # 4) Ablation: labeled data size
    lbl_cfg = ps.get("ablations", {}).get("labeled_data_size", {})
    if lbl_cfg.get("enabled", False):
        subset_dir = ensure_dir(run_root / "manifests_subsets")
        seed = int(lbl_cfg.get("seed", 42))
        for frac in lbl_cfg.get("fractions", [1.0]):
            frac = float(frac)
            subset_manifest = subset_dir / f"labeled_train_frac_{str(frac).replace('.', 'p')}.jsonl"
            n_rows = _subset_manifest(train_manifest, subset_manifest, fraction=frac, seed=seed)
            row = run_finetune_and_validate(
                base_cfg=cfg,
                patch={"finetune": {"pretrained_checkpoint": str(pretrained_ckpt_path)}},
                run_id=f"ablation_labeled_frac_{str(frac).replace('.', 'p')}",
                run_root=run_root,
                train_manifest=subset_manifest,
                val_manifest=val_manifest,
            )
            row["fraction"] = frac
            row["train_samples"] = int(n_rows)
            summary_rows["ablation_labeled_data_size"].append(row)

    # Choose a checkpoint for inference/runtime/robustness (prefer pretrained baseline run, else configured semantic checkpoint).
    semantic_ckpt = abs_path(ps["assets"]["semantic_checkpoint"])
    if summary_rows["scratch_vs_pretrained"]:
        for r in summary_rows["scratch_vs_pretrained"]:
            if r.get("mode") == "contrastive_pretrained":
                semantic_ckpt = Path(r["checkpoint"])
                break

    # 5) Ablation: temporal smoothing (inference-time)
    ts_cfg = ps.get("ablations", {}).get("temporal_smoothing", {})
    runtime_frames = int(ps.get("runtime", {}).get("frames", 20))
    if ts_cfg.get("enabled", False):
        for window in ts_cfg.get("windows", [1, 5]):
            row = benchmark_inference_runtime(
                cfg=cfg,
                checkpoint_path=semantic_ckpt,
                input_path=unlabeled_root,
                frames=runtime_frames,
                temporal_window=int(window),
            )
            row["window"] = int(window)
            summary_rows["ablation_temporal_smoothing"].append(row)

    # 6) Runtime benchmark
    rt_cfg = ps.get("runtime", {})
    if rt_cfg.get("enabled", False):
        if rt_cfg.get("sweep", "voxel_size") == "voxel_size":
            for voxel_size in rt_cfg.get("voxel_sizes", [cfg["data"]["voxel_size"]]):
                rt_cfg_local = _deep_update(cfg, {"data": {"voxel_size": float(voxel_size)}})
                row = benchmark_inference_runtime(
                    cfg=rt_cfg_local,
                    checkpoint_path=semantic_ckpt,
                    input_path=unlabeled_root,
                    frames=runtime_frames,
                    temporal_window=1,
                )
                row["voxel_size"] = float(voxel_size)
                summary_rows["runtime_benchmark"].append(row)

    # 7) Robustness evaluation on validation set
    rb_cfg = ps.get("robustness", {})
    if rb_cfg.get("enabled", False):
        np.random.seed(int(cfg.get("seed", 42)))
        clean = evaluate_robustness(cfg=cfg, checkpoint_path=semantic_ckpt, manifest_path=val_manifest, mode="clean")
        for mode in ["clean", "noise", "occlusion", "color_degradation"]:
            metrics = clean if mode == "clean" else evaluate_robustness(
                cfg=cfg,
                checkpoint_path=semantic_ckpt,
                manifest_path=val_manifest,
                mode=mode,
            )
            row = {
                "condition": mode,
                "miou": float(metrics["miou"]),
                "accuracy": float(metrics["accuracy"]),
                "miou_drop_vs_clean": float(clean["miou"] - metrics["miou"]),
                "accuracy_drop_vs_clean": float(clean["accuracy"] - metrics["accuracy"]),
            }
            for cname, payload in metrics["per_class"].items():
                row[f"iou_{cname}"] = float(payload["iou"])
            summary_rows["robustness"].append(row)

    # Persist summary tables
    summary_dir = ensure_dir(run_root / "summaries")
    for key, rows in summary_rows.items():
        write_json(summary_dir / f"{key}.json", {"rows": rows})
        _save_csv(summary_dir / f"{key}.csv", rows)

    print("[Study] Summary tables written:")
    for key in summary_rows.keys():
        print(f"  - {summary_dir / (key + '.csv')}")

    if not args.no_visualize:
        _run_cmd(
            [
                sys.executable,
                "MinkUNET/scripts/visualize_paper_results.py",
                "--study-dir",
                str(run_root),
            ],
            cwd=REPO_ROOT,
        )

    print(f"[Done] Paper study complete: {run_root}")


if __name__ == "__main__":
    main()
