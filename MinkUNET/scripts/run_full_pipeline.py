from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(args: list[str]) -> None:
    cmd = [sys.executable] + args
    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate MinkUNET segmentation workflow.")
    parser.add_argument("--stage", choices=["setup", "train", "all"], default="setup")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--base-config", default="MinkUNET/configs/base.yaml")
    parser.add_argument("--pretrain-config", default="MinkUNET/configs/pretrain.yaml")
    parser.add_argument("--finetune-config", default="MinkUNET/configs/finetune.yaml")
    parser.add_argument("--inference-config", default="MinkUNET/configs/inference.yaml")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    args = parser.parse_args()

    script_create_sample = str(ROOT / "create_annotation_sample.py")
    script_cluster = str(ROOT / "cluster_for_annotation.py")
    script_apply_labels = str(ROOT / "apply_cluster_labels.py")
    script_manifests = str(ROOT / "build_manifests.py")
    script_train_ssl = str(ROOT / "train_contrastive.py")
    script_train_seg = str(ROOT / "train_segmentation.py")
    script_validate = str(ROOT / "validate_segmentation.py")
    script_infer = str(ROOT / "infer_segmentation.py")

    if args.stage in ("setup", "all"):
        run_step([script_create_sample, "--config", args.base_config, "--count", str(args.sample_count)])
        run_step([script_cluster, "--config", args.base_config])
        run_step([script_manifests, "--config", args.base_config])
        print(
            "[Manual Step] Fill manual_label fields in "
            "Datasets/Annotated/cluster_proposals/*_cluster_labels.json before training."
        )

    if args.stage in ("train", "all"):
        run_step([script_apply_labels, "--config", args.base_config])
        run_step([script_manifests, "--config", args.base_config])
        if not args.skip_pretrain:
            run_step([script_train_ssl, "--config", args.pretrain_config])
        run_step([script_train_seg, "--config", args.finetune_config])
        run_step([script_validate, "--config", args.finetune_config])
        if not args.skip_infer:
            run_step([script_infer, "--config", args.inference_config])


if __name__ == "__main__":
    main()

