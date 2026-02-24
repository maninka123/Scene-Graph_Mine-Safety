from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import abs_path
from minkunet.utils.io import ensure_dir, write_json

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _plot_bar(path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    if not labels:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_line(path: Path, x: List[float], y: List[float], title: str, xlabel: str, ylabel: str) -> None:
    if not x:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _latest_study_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No study directories found in {root}")
    return dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper comparison plots from run_paper_study summaries.")
    parser.add_argument("--study-dir", default=None, help="Result folder like Results/Paper_Study/<run_name_timestamp>")
    parser.add_argument("--study-root", default="Results/Paper_Study")
    args = parser.parse_args()

    if args.study_dir:
        study_dir = abs_path(args.study_dir)
    else:
        study_dir = _latest_study_dir(abs_path(args.study_root))

    if plt is None:
        raise RuntimeError(
            "matplotlib import failed in this environment. Activate the project venv and reinstall matplotlib."
        )

    summary_dir = study_dir / "summaries"
    plot_dir = ensure_dir(study_dir / "plots")

    # 1) Scratch vs pretrained
    scratch_rows = _read_csv(summary_dir / "scratch_vs_pretrained.csv")
    if scratch_rows:
        labels = [r.get("mode", "unknown") for r in scratch_rows]
        miou = [_to_float(r.get("miou", "0")) for r in scratch_rows]
        acc = [_to_float(r.get("accuracy", "0")) for r in scratch_rows]
        _plot_bar(plot_dir / "scratch_vs_pretrained_miou.png", labels, miou, "Scratch vs Pretrained (mIoU)", "mIoU")
        _plot_bar(plot_dir / "scratch_vs_pretrained_accuracy.png", labels, acc, "Scratch vs Pretrained (Accuracy)", "Accuracy")

    # 2) Ablation: voxel size
    vox_rows = _read_csv(summary_dir / "ablation_voxel_size.csv")
    if vox_rows:
        xs = [_to_float(r.get("voxel_size", "0")) for r in vox_rows]
        ys = [_to_float(r.get("miou", "0")) for r in vox_rows]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        _plot_line(
            plot_dir / "ablation_voxel_size_miou.png",
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            "Ablation: Voxel Size vs mIoU",
            "Voxel Size (m)",
            "mIoU",
        )

    # 3) Ablation: RGB on/off
    rgb_rows = _read_csv(summary_dir / "ablation_rgb.csv")
    if rgb_rows:
        labels = ["RGB on" if r.get("use_rgb", "").lower() == "true" else "RGB off" for r in rgb_rows]
        ys = [_to_float(r.get("miou", "0")) for r in rgb_rows]
        _plot_bar(plot_dir / "ablation_rgb_miou.png", labels, ys, "Ablation: RGB On/Off", "mIoU")

    # 4) Ablation: labeled data size
    lbl_rows = _read_csv(summary_dir / "ablation_labeled_data_size.csv")
    if lbl_rows:
        xs = [_to_float(r.get("fraction", "0")) for r in lbl_rows]
        ys = [_to_float(r.get("miou", "0")) for r in lbl_rows]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        _plot_line(
            plot_dir / "ablation_labeled_fraction_miou.png",
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            "Ablation: Labeled Data Fraction vs mIoU",
            "Labeled Fraction",
            "mIoU",
        )

    # 5) Temporal smoothing
    ts_rows = _read_csv(summary_dir / "ablation_temporal_smoothing.csv")
    if ts_rows:
        labels = [f"window={int(_to_float(r.get('window', '1')))}" for r in ts_rows]
        fps = [_to_float(r.get("fps", "0")) for r in ts_rows]
        stability = [_to_float(r.get("stability_score", "0")) for r in ts_rows]
        _plot_bar(plot_dir / "ablation_temporal_fps.png", labels, fps, "Temporal Smoothing vs FPS", "FPS")
        _plot_bar(plot_dir / "ablation_temporal_stability.png", labels, stability, "Temporal Smoothing vs Stability", "Stability Score")

    # 6) Runtime benchmark
    rt_rows = _read_csv(summary_dir / "runtime_benchmark.csv")
    if rt_rows:
        xs = [_to_float(r.get("voxel_size", "0")) for r in rt_rows]
        fps = [_to_float(r.get("fps", "0")) for r in rt_rows]
        lat = [_to_float(r.get("avg_latency_sec", "0")) * 1000.0 for r in rt_rows]
        mem = [_to_float(r.get("gpu_memory_mb_peak", "0")) for r in rt_rows]
        pairs_fps = sorted(zip(xs, fps), key=lambda t: t[0])
        pairs_lat = sorted(zip(xs, lat), key=lambda t: t[0])
        pairs_mem = sorted(zip(xs, mem), key=lambda t: t[0])
        _plot_line(plot_dir / "runtime_voxel_vs_fps.png", [p[0] for p in pairs_fps], [p[1] for p in pairs_fps], "Runtime: Voxel Size vs FPS", "Voxel Size (m)", "FPS")
        _plot_line(plot_dir / "runtime_voxel_vs_latency_ms.png", [p[0] for p in pairs_lat], [p[1] for p in pairs_lat], "Runtime: Voxel Size vs Latency", "Voxel Size (m)", "Latency (ms)")
        _plot_line(plot_dir / "runtime_voxel_vs_gpu_mem_mb.png", [p[0] for p in pairs_mem], [p[1] for p in pairs_mem], "Runtime: Voxel Size vs GPU Memory", "Voxel Size (m)", "GPU Memory (MB)")

    # 7) Robustness
    rb_rows = _read_csv(summary_dir / "robustness.csv")
    if rb_rows:
        labels = [r.get("condition", "unknown") for r in rb_rows]
        miou = [_to_float(r.get("miou", "0")) for r in rb_rows]
        drop = [_to_float(r.get("miou_drop_vs_clean", "0")) for r in rb_rows]
        _plot_bar(plot_dir / "robustness_miou.png", labels, miou, "Robustness: mIoU by Condition", "mIoU")
        _plot_bar(plot_dir / "robustness_miou_drop.png", labels, drop, "Robustness: mIoU Drop vs Clean", "mIoU Drop")

    index = {
        "study_dir": str(study_dir),
        "summary_dir": str(summary_dir),
        "plot_dir": str(plot_dir),
    }
    write_json(study_dir / "plots" / "plot_index.json", index)
    print(f"[Done] Paper plots written to {plot_dir}")


if __name__ == "__main__":
    main()
