from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _common import abs_path, load_cfg
from minkunet.data.collate import collate_supervised
from minkunet.data.datasets import LabeledNPZDataset
from minkunet.engine.trainers import train_segmentation
from minkunet.models.network import SemanticMinkUNet
from minkunet.utils.seed import set_global_seed


def _manifest_row_count(path_value: str) -> int:
    path = Path(path_value)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune semantic segmentation head on labeled samples.")
    parser.add_argument("--config", default="MinkUNET/configs/finetune.yaml")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_global_seed(int(cfg.get("seed", 42)))

    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    manifests_dir = abs_path(cfg["data"]["manifests_dir"])
    train_manifest = args.train_manifest or str(manifests_dir / "labeled_train.jsonl")
    val_manifest = args.val_manifest or str(manifests_dir / "labeled_val.jsonl")

    train_rows = _manifest_row_count(train_manifest)
    val_rows = _manifest_row_count(val_manifest)
    if train_rows == 0:
        print(f"[Error] Missing labeled manifests: train_rows={train_rows}, val_rows={val_rows}")
        print("[Next] Generate labels then manifests:")
        print("  python MinkUNET/scripts/apply_cluster_labels.py")
        print("  python MinkUNET/scripts/build_manifests.py")
        print("  # Ensure *_cluster_labels.json files have manual_label filled before that.")
        sys.exit(1)
    if val_rows == 0:
        print(
            f"[Warn] Validation manifest is empty ({val_manifest}). "
            "Using train manifest for validation due to small labeled set."
        )
        val_manifest = train_manifest

    train_dataset = LabeledNPZDataset(
        manifest_path=train_manifest,
        cfg=cfg,
        train=True,
        augmentation_cfg=cfg["finetune"]["augmentation"],
    )
    val_dataset = LabeledNPZDataset(
        manifest_path=val_manifest,
        cfg=cfg,
        train=False,
        augmentation_cfg=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_supervised,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
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

    best_ckpt = train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    print(f"[Done] Best semantic checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
