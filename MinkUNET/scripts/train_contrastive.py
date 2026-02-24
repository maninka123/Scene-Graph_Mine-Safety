from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from _common import abs_path, load_cfg
from minkunet.data.collate import collate_contrastive
from minkunet.data.datasets import UnlabeledContrastiveDataset
from minkunet.engine.trainers import train_contrastive
from minkunet.models.network import ContrastiveMinkUNet
from minkunet.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for MinkUNET backbone.")
    parser.add_argument("--config", default="MinkUNET/configs/pretrain.yaml")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--skip-val", action="store_true", help="Skip validation loop during pretraining.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_global_seed(int(cfg.get("seed", 42)))

    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    manifests_dir = abs_path(cfg["data"]["manifests_dir"])
    train_manifest = args.train_manifest or str(manifests_dir / "unlabeled_train.jsonl")
    val_manifest = args.val_manifest or str(manifests_dir / "unlabeled_val.jsonl")

    train_dataset = UnlabeledContrastiveDataset(
        manifest_path=train_manifest,
        cfg=cfg,
        augmentation_cfg=cfg["pretrain"]["augmentation"],
    )
    val_dataset = None
    if not args.skip_val:
        val_dataset = UnlabeledContrastiveDataset(
            manifest_path=val_manifest,
            cfg=cfg,
            augmentation_cfg=cfg["pretrain"]["augmentation"],
        )

    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_contrastive,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_contrastive,
            drop_last=True,
        )

    try:
        model = ContrastiveMinkUNet(
            in_channels=int(cfg["model"]["in_channels"]),
            feature_dim=int(cfg["model"]["feature_dim"]),
            projection_dim=int(cfg["pretrain"]["projection_dim"]),
            width=int(cfg["model"]["width"]),
        ).to(device)
    except ImportError as exc:
        print(str(exc))
        sys.exit(1)

    best_ckpt = train_contrastive(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    print(f"[Done] Best contrastive checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
