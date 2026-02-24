from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common import abs_path, load_cfg
from minkunet.data.collate import collate_supervised
from minkunet.data.datasets import LabeledNPZDataset
from minkunet.engine.checkpoint import load_checkpoint
from minkunet.engine.eval import evaluate_segmentation
from minkunet.models.network import SemanticMinkUNet

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate semantic segmentation checkpoint.")
    parser.add_argument("--config", default="MinkUNET/configs/finetune.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--save-json", default="MinkUNET/logs/validation_metrics.json")
    parser.add_argument("--output-dir", default=None, help="Optional directory for validation artifacts.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    manifests_dir = abs_path(cfg["data"]["manifests_dir"])
    val_manifest = args.manifest or str(manifests_dir / "labeled_val.jsonl")
    if args.checkpoint:
        checkpoint_path = abs_path(args.checkpoint)
    else:
        checkpoint_name = cfg["finetune"].get("checkpoint_name", "semantic_best.pt")
        checkpoint_path = Path(checkpoint_name)
        if not checkpoint_path.is_absolute() and len(checkpoint_path.parts) == 1:
            checkpoint_path = Path(cfg["checkpoint"]["dir"]) / checkpoint_path
        checkpoint_path = abs_path(checkpoint_path)
    checkpoint_path = str(checkpoint_path)

    dataset = LabeledNPZDataset(
        manifest_path=val_manifest,
        cfg=cfg,
        train=False,
        augmentation_cfg=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_supervised,
        drop_last=False,
    )

    try:
        model = SemanticMinkUNet(
            in_channels=int(cfg["model"]["in_channels"]),
            feature_dim=int(cfg["model"]["feature_dim"]),
            num_classes=int(cfg["model"]["num_classes"]),
            width=int(cfg["model"]["width"]),
        ).to(device)
    except ImportError as exc:
        print(str(exc))
        sys.exit(1)

    load_checkpoint(checkpoint_path, model=model, map_location=str(device))

    class_weights = torch.tensor(
        cfg["finetune"].get("class_weights", [1.0] * int(cfg["model"]["num_classes"])),
        dtype=torch.float32,
        device=device,
    )
    ignore_index = int(cfg["finetune"].get("ignore_index", -1))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    metrics = evaluate_segmentation(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=int(cfg["model"]["num_classes"]),
        class_names=cfg["model"]["class_names"],
        ignore_index=ignore_index,
        return_hist=True,
    )
    confusion_matrix = metrics.pop("confusion_matrix", None)

    print(json.dumps(metrics, indent=2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = abs_path(args.output_dir) if args.output_dir else abs_path(f"MinkUNET/logs/validation/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if plt is not None:
        class_names = [str(name) for name in cfg["model"]["class_names"]]
        class_ious = [float(metrics["per_class"].get(name, {}).get("iou", 0.0)) for name in class_names]
        plt.figure(figsize=(9, 5))
        plt.bar(class_names, class_ious)
        plt.title("Validation Per-Class IoU")
        plt.xlabel("Class")
        plt.ylabel("IoU")
        plt.ylim(0.0, 1.0)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "per_class_iou.png", dpi=160)
        plt.close()

    if confusion_matrix is not None:
        cm = np.asarray(confusion_matrix, dtype=np.int64)
        np.save(output_dir / "confusion_matrix.npy", cm)
        with (output_dir / "confusion_matrix.json").open("w", encoding="utf-8") as handle:
            json.dump({"class_names": cfg["model"]["class_names"], "matrix": cm.tolist()}, handle, indent=2)

        if plt is not None:
            class_names = [str(name) for name in cfg["model"]["class_names"]]
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation="nearest")
            ax.figure.colorbar(im, ax=ax)
            ax.set_title("Confusion Matrix (Counts)")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Ground Truth Class")
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)

            max_val = int(cm.max()) if cm.size else 0
            threshold = max_val * 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = int(cm[i, j])
                    color = "white" if val > threshold else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix_counts.png", dpi=180)
            plt.close(fig)

            # Row-normalized heatmap for easier class-wise confusion analysis.
            row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
            row_sum[row_sum == 0] = 1.0
            cm_norm = cm.astype(np.float64) / row_sum

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            im2 = ax2.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
            ax2.figure.colorbar(im2, ax=ax2)
            ax2.set_title("Confusion Matrix (Row Normalized)")
            ax2.set_xlabel("Predicted Class")
            ax2.set_ylabel("Ground Truth Class")
            ax2.set_xticks(np.arange(len(class_names)))
            ax2.set_yticks(np.arange(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha="right")
            ax2.set_yticklabels(class_names)

            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    val = float(cm_norm[i, j])
                    color = "white" if val > 0.5 else "black"
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

            fig2.tight_layout()
            fig2.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
            plt.close(fig2)

    # Backward-compatible single file output.
    out_json = abs_path(args.save_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"[Done] Saved validation artifacts to {output_dir}")
    print(f"[Done] Saved metrics JSON (legacy path) to {out_json}")


if __name__ == "__main__":
    main()
