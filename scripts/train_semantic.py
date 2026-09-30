from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mine_safety.config import load_config
from mine_safety.perception.data import NpzPointCloudDataset, sparse_sample
from mine_safety.perception.network import SemanticMinkUNet, require_minkowski


def main() -> None:
    parser = argparse.ArgumentParser(description="Nine-class supervised fine-tuning")
    parser.add_argument("manifest", type=Path, help="Training JSONL manifest")
    parser.add_argument("--val-manifest", type=Path, required=True, help="Held-out validation JSONL manifest")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "checkpoints" / "contrastive_backbone.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "semantic_nine_class.pt")
    args = parser.parse_args()
    cfg = load_config()
    pcfg, train = cfg["perception"], cfg["perception"]["finetune"]
    dataset = NpzPointCloudDataset(args.manifest)
    validation = NpzPointCloudDataset(args.val_manifest)
    if not dataset or not validation:
        parser.error("training and validation manifests must both contain samples")
    train_paths = {record["path"].resolve() for record in dataset.records}
    val_paths = {record["path"].resolve() for record in validation.records}
    if train_paths & val_paths:
        parser.error("training and validation manifests must not share point clouds")
    loader = DataLoader(dataset, batch_size=train["batch_size"], shuffle=True, collate_fn=lambda batch: batch)
    val_loader = DataLoader(validation, batch_size=train["batch_size"], shuffle=False, collate_fn=lambda batch: batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticMinkUNet(
        in_channels=pcfg["input_features"], width=pcfg["backbone_width"],
        feature_dim=pcfg["feature_dim"], num_classes=len(pcfg["classes"]),
    ).to(device)
    if args.pretrained.exists():
        saved = torch.load(args.pretrained, map_location="cpu")
        state = saved.get("model_state_dict", saved)
        backbone = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
        model.backbone.load_state_dict(backbone, strict=True)
    optimiser = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": train["backbone_learning_rate"]},
        {"params": model.head.parameters(), "lr": train["head_learning_rate"]},
    ])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=train["epochs"])
    me = require_minkowski()
    best_loss = float("inf")
    patience, stale = 12, 0
    def mean_loss(data_loader: DataLoader, *, training: bool) -> float:
        model.train(training)
        total = 0.0
        for batch in data_loader:
            samples = [sparse_sample(item["points"], item["colors"], pcfg["voxel_size_m"], item["labels"]) for item in batch]
            coordinates, features, labels = me.utils.sparse_collate(
                [item["coordinates"] for item in samples], [item["features"] for item in samples],
                [item["labels"] for item in samples],
            )
            sparse = me.SparseTensor(features.float().to(device), coordinates=coordinates.to(device))
            loss = torch.nn.functional.cross_entropy(model(sparse).F, labels.long().to(device))
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            total += float(loss.detach())
        return total / len(data_loader)

    for epoch in range(train["epochs"]):
        train_loss = mean_loss(loader, training=True)
        with torch.no_grad():
            val_loss = mean_loss(val_loader, training=False)
        schedule.step()
        print(f"epoch={epoch + 1:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, args.output)
        else:
            stale += 1
            if train["early_stopping"] and stale >= patience:
                print("early stopping")
                break


if __name__ == "__main__":
    main()
