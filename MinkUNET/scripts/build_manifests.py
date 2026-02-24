from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from _common import REPO_ROOT, abs_path, load_cfg
from minkunet.utils.io import (
    ensure_dir,
    read_jsonl,
    resolve_existing_dir,
    write_json,
    write_jsonl,
)


def split_records(records: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not records:
        return [], []
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    val_count = max(1, int(len(records) * val_ratio)) if len(records) > 1 else 0
    val_indices = set(indices[:val_count])
    train = [records[i] for i in range(len(records)) if i not in val_indices]
    val = [records[i] for i in range(len(records)) if i in val_indices]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val manifests for MinkUNET pipeline.")
    parser.add_argument("--config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlabeled-val-ratio", type=float, default=0.1)
    parser.add_argument("--labeled-val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    manifests_dir = ensure_dir(abs_path(data_cfg["manifests_dir"]))

    unlabeled_root = resolve_existing_dir(data_cfg["unlabeled_roots"], root=REPO_ROOT)
    unlabeled_records = [{"pcd_path": str(p.resolve())} for p in sorted(unlabeled_root.glob("*.pcd"))]
    unlabeled_train, unlabeled_val = split_records(
        unlabeled_records,
        val_ratio=float(args.unlabeled_val_ratio),
        seed=args.seed,
    )
    write_jsonl(manifests_dir / "unlabeled_train.jsonl", unlabeled_train)
    write_jsonl(manifests_dir / "unlabeled_val.jsonl", unlabeled_val)

    labels_dir = abs_path(data_cfg["labels_dir"])
    labeled_records = [{"label_npz": str(p.resolve())} for p in sorted(labels_dir.glob("*_labels.npz"))]

    labeled_manifest_all = manifests_dir / "labeled_all.jsonl"
    if labeled_manifest_all.exists():
        manifest_rows = read_jsonl(labeled_manifest_all)
        if manifest_rows:
            labeled_records = manifest_rows

    labeled_train, labeled_val = split_records(
        labeled_records,
        val_ratio=float(args.labeled_val_ratio),
        seed=args.seed,
    )
    write_jsonl(manifests_dir / "labeled_train.jsonl", labeled_train)
    write_jsonl(manifests_dir / "labeled_val.jsonl", labeled_val)

    summary = {
        "unlabeled_root": str(unlabeled_root),
        "unlabeled_total": len(unlabeled_records),
        "unlabeled_train": len(unlabeled_train),
        "unlabeled_val": len(unlabeled_val),
        "labeled_total": len(labeled_records),
        "labeled_train": len(labeled_train),
        "labeled_val": len(labeled_val),
        "manifests_dir": str(manifests_dir),
    }
    write_json(manifests_dir / "manifest_summary.json", summary)

    print("[Manifest] written:")
    for name in (
        "unlabeled_train.jsonl",
        "unlabeled_val.jsonl",
        "labeled_train.jsonl",
        "labeled_val.jsonl",
    ):
        p = manifests_dir / name
        print(f"  - {p}")
    print(f"[Manifest] summary={manifests_dir / 'manifest_summary.json'}")


if __name__ == "__main__":
    main()

