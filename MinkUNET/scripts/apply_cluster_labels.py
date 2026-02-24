from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import abs_path, load_cfg
from minkunet.utils.io import ensure_dir, read_json, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert manual cluster labels into per-point labels.")
    parser.add_argument("--config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--strict", action="store_true", help="Fail on unknown manual labels.")
    parser.add_argument("--min-labeled-ratio", type=float, default=0.01)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    data_cfg = cfg["data"]
    class_names = cfg["model"]["class_names"]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    proposals_dir = abs_path(data_cfg["cluster_proposals_dir"])
    labels_dir = ensure_dir(abs_path(data_cfg["labels_dir"]))
    manifests_dir = ensure_dir(abs_path(data_cfg["manifests_dir"]))

    template_files = sorted(proposals_dir.glob("*_cluster_labels.json"))
    if not template_files:
        raise RuntimeError(f"No cluster label templates found in {proposals_dir}")

    manifest_rows = []
    report = []

    for template_path in template_files:
        template = read_json(template_path)
        proposal_npz = Path(template["proposal_npz"])
        if not proposal_npz.exists():
            raise FileNotFoundError(f"Missing proposal NPZ: {proposal_npz}")

        payload = np.load(proposal_npz, allow_pickle=True)
        points = payload["points"].astype(np.float32)
        colors = payload["colors"].astype(np.float32)
        cluster_ids = payload["cluster_ids"].astype(np.int64)

        point_labels = np.full(cluster_ids.shape[0], -1, dtype=np.int64)
        unknown_labels = []

        for cluster_item in template["clusters"]:
            manual_label = str(cluster_item.get("manual_label", "")).strip().lower()
            if not manual_label:
                continue
            if manual_label not in class_to_id:
                unknown_labels.append(manual_label)
                continue
            cluster_id = int(cluster_item["cluster_id"])
            point_labels[cluster_ids == cluster_id] = class_to_id[manual_label]

        if unknown_labels and args.strict:
            unknown = sorted(set(unknown_labels))
            raise ValueError(f"{template_path.name} has unknown labels: {unknown}")

        labeled_ratio = float((point_labels >= 0).sum() / max(point_labels.shape[0], 1))
        if labeled_ratio < args.min_labeled_ratio:
            print(
                f"[Skip] {template_path.name}: labeled_ratio={labeled_ratio:.3f} < {args.min_labeled_ratio:.3f}"
            )
            continue

        out_npz = labels_dir / template_path.name.replace("_cluster_labels.json", "_labels.npz")
        np.savez_compressed(
            out_npz,
            points=points,
            colors=colors,
            labels=point_labels,
            class_names=np.array(class_names),
            source_pcd=np.array([template.get("source_pcd", "")]),
        )

        manifest_rows.append({"label_npz": str(out_npz.resolve())})
        report.append(
            {
                "template_file": str(template_path.resolve()),
                "output_label_npz": str(out_npz.resolve()),
                "total_points": int(point_labels.shape[0]),
                "labeled_points": int((point_labels >= 0).sum()),
                "labeled_ratio": labeled_ratio,
            }
        )

        print(
            f"[Labels] {template_path.stem}: labeled_points={(point_labels >= 0).sum()} "
            f"total={point_labels.shape[0]} ratio={labeled_ratio:.3f}"
        )

    labeled_manifest = manifests_dir / "labeled_all.jsonl"
    write_jsonl(labeled_manifest, manifest_rows)
    write_json(manifests_dir / "label_conversion_report.json", {"items": report})

    print(f"[Done] labeled manifest: {labeled_manifest}")
    print(f"[Done] labels dir: {labels_dir}")


if __name__ == "__main__":
    main()

