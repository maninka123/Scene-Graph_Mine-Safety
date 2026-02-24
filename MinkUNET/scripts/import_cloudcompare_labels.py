from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from _common import abs_path, load_cfg
from minkunet.data.pcd import read_pcd
from minkunet.utils.io import ensure_dir, write_json, write_jsonl


def _quantize_points(points: np.ndarray, tol: float) -> np.ndarray:
    return np.round(points / tol).astype(np.int64)


def _build_index(points: np.ndarray, tol: float) -> Dict[Tuple[int, int, int], List[int]]:
    quant = _quantize_points(points, tol=tol)
    index: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(quant.shape[0]):
        key = (int(quant[i, 0]), int(quant[i, 1]), int(quant[i, 2]))
        index.setdefault(key, []).append(i)
    return index


def _assign_class_points(
    source_points: np.ndarray,
    labels: np.ndarray,
    class_points: np.ndarray,
    class_id: int,
    point_index: Dict[Tuple[int, int, int], List[int]],
    tol: float,
) -> Tuple[int, int]:
    assigned = 0
    unresolved = 0
    quant = _quantize_points(class_points, tol=tol)

    for i in range(quant.shape[0]):
        key = (int(quant[i, 0]), int(quant[i, 1]), int(quant[i, 2]))
        candidates = point_index.get(key)
        if not candidates:
            unresolved += 1
            continue

        if len(candidates) == 1:
            idx = candidates[0]
        else:
            p = class_points[i]
            candidate_points = source_points[candidates]
            idx = candidates[int(np.argmin(np.sum((candidate_points - p) ** 2, axis=1)))]

        labels[idx] = class_id
        assigned += 1

    return assigned, unresolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import CloudCompare per-class PCD annotations from manual_cc into label NPZ files."
    )
    parser.add_argument("--config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--min-labeled-ratio", type=float, default=0.01)
    parser.add_argument("--point-match-tol", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    class_names = [str(name).strip().lower() for name in cfg["model"]["class_names"]]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    sample_dir = abs_path(data_cfg["sample_dir"])
    manual_cc_dir = abs_path(data_cfg.get("manual_cc_dir", Path(data_cfg["annotated_root"]) / "manual_cc"))
    labels_dir = ensure_dir(abs_path(data_cfg["labels_dir"]))
    manifests_dir = ensure_dir(abs_path(data_cfg["manifests_dir"]))

    if not manual_cc_dir.exists():
        raise FileNotFoundError(f"manual_cc folder not found: {manual_cc_dir}")

    frame_dirs = sorted([p for p in manual_cc_dir.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise RuntimeError(f"No frame folders found under: {manual_cc_dir}")

    manifest_rows = []
    report_items = []

    for frame_dir in frame_dirs:
        frame_name = frame_dir.name
        source_pcd = sample_dir / f"{frame_name}.pcd"
        if not source_pcd.exists():
            print(f"[Skip] Missing source sample PCD for frame: {frame_name}")
            continue

        points, colors = read_pcd(source_pcd)
        labels = np.full(points.shape[0], -1, dtype=np.int64)
        point_index = _build_index(points, tol=float(args.point_match_tol))

        class_stats = {}
        total_assigned = 0
        total_unresolved = 0

        for class_name, class_id in class_to_id.items():
            class_file = frame_dir / f"{class_name}.pcd"
            if not class_file.exists():
                class_stats[class_name] = {"exists": False, "assigned": 0, "unresolved": 0}
                continue

            class_points, _ = read_pcd(class_file)
            assigned, unresolved = _assign_class_points(
                source_points=points,
                labels=labels,
                class_points=class_points,
                class_id=class_id,
                point_index=point_index,
                tol=float(args.point_match_tol),
            )
            total_assigned += assigned
            total_unresolved += unresolved
            class_stats[class_name] = {
                "exists": True,
                "input_points": int(class_points.shape[0]),
                "assigned": int(assigned),
                "unresolved": int(unresolved),
            }

        labeled_ratio = float((labels >= 0).sum() / max(labels.shape[0], 1))
        if labeled_ratio < args.min_labeled_ratio:
            print(f"[Skip] {frame_name}: labeled_ratio={labeled_ratio:.3f} < {args.min_labeled_ratio:.3f}")
            continue

        out_npz = labels_dir / f"{frame_name}_labels.npz"
        np.savez_compressed(
            out_npz,
            points=points.astype(np.float32),
            colors=colors.astype(np.float32),
            labels=labels,
            class_names=np.array(class_names),
            source_pcd=np.array([str(source_pcd.resolve())]),
            source_annotation_dir=np.array([str(frame_dir.resolve())],
                                           dtype=object),
        )

        manifest_rows.append({"label_npz": str(out_npz.resolve())})
        report_items.append(
            {
                "frame": frame_name,
                "source_pcd": str(source_pcd.resolve()),
                "annotation_dir": str(frame_dir.resolve()),
                "output_label_npz": str(out_npz.resolve()),
                "total_points": int(points.shape[0]),
                "labeled_points": int((labels >= 0).sum()),
                "labeled_ratio": labeled_ratio,
                "total_assigned_from_class_files": int(total_assigned),
                "total_unresolved_from_class_files": int(total_unresolved),
                "class_stats": class_stats,
            }
        )

        print(
            f"[Import] {frame_name}: labeled_points={(labels >= 0).sum()} "
            f"total={labels.shape[0]} ratio={labeled_ratio:.3f}"
        )

    labeled_manifest = manifests_dir / "labeled_all.jsonl"
    write_jsonl(labeled_manifest, manifest_rows)
    write_json(manifests_dir / "manual_cc_import_report.json", {"items": report_items})

    print(f"[Done] labeled manifest: {labeled_manifest}")
    print(f"[Done] labels dir: {labels_dir}")
    print(f"[Done] report: {manifests_dir / 'manual_cc_import_report.json'}")


if __name__ == "__main__":
    main()

