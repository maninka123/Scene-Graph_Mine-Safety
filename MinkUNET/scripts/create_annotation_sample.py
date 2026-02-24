from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from _common import REPO_ROOT, abs_path, load_cfg
from minkunet.utils.io import ensure_dir, resolve_existing_dir, write_jsonl


def _uniform_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    step = total / float(count)
    return sorted({min(int(i * step), total - 1) for i in range(count)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small sample set for manual annotation.")
    parser.add_argument("--config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--strategy", choices=["uniform", "random"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]

    unlabeled_root = resolve_existing_dir(data_cfg["unlabeled_roots"], root=REPO_ROOT)
    sample_dir = ensure_dir(abs_path(data_cfg["sample_dir"]))
    manual_cc_dir = ensure_dir(
        abs_path(data_cfg.get("manual_cc_dir", Path(data_cfg["annotated_root"]) / "manual_cc"))
    )
    manifests_dir = ensure_dir(abs_path(data_cfg["manifests_dir"]))

    pcd_files = sorted(unlabeled_root.glob("*.pcd"))
    if not pcd_files:
        raise RuntimeError(f"No PCD files found in {unlabeled_root}")

    count = min(args.count, len(pcd_files))
    if args.strategy == "uniform":
        selected_indices = _uniform_indices(len(pcd_files), count)
    else:
        random.seed(args.seed)
        selected_indices = sorted(random.sample(range(len(pcd_files)), k=count))
    selected_files = [pcd_files[idx] for idx in selected_indices]

    copied = 0
    annotation_dirs_created = 0
    for src in selected_files:
        dst = sample_dir / src.name
        if dst.exists() and not args.overwrite:
            pass
        else:
            shutil.copy2(src, dst)
            copied += 1

        frame_annotation_dir = manual_cc_dir / src.stem
        if not frame_annotation_dir.exists():
            frame_annotation_dir.mkdir(parents=True, exist_ok=True)
            annotation_dirs_created += 1

    manifest_rows = [
        {
            "pcd_path": str((sample_dir / f.name).resolve()),
            "annotation_dir": str((manual_cc_dir / f.stem).resolve()),
        }
        for f in selected_files
    ]
    manifest_path = manifests_dir / "sample_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    print(f"[Sample] source={unlabeled_root}")
    print(f"[Sample] sample_dir={sample_dir}")
    print(f"[Sample] manual_cc_dir={manual_cc_dir}")
    print(f"[Sample] selected={len(selected_files)} copied={copied}")
    print(f"[Sample] annotation_dirs_created={annotation_dirs_created}")
    print(f"[Sample] manifest={manifest_path}")


if __name__ == "__main__":
    main()
