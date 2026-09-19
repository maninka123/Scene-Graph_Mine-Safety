from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mine_safety.config import load_config
from mine_safety.perception.data import NpzPointCloudDataset, augment_points, sparse_sample
from mine_safety.perception.network import ContrastiveMinkUNet, nt_xent_loss, require_minkowski


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned NT-Xent pretraining")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "contrastive_backbone.pt")
    args = parser.parse_args()
    cfg = load_config()
    pcfg, train = cfg["perception"], cfg["perception"]["contrastive"]
    dataset = NpzPointCloudDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=train["batch_size"], shuffle=True, collate_fn=lambda batch: batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveMinkUNet(pcfg["input_features"], pcfg["feature_dim"], pcfg["projection_dim"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=train["epochs"])
    me = require_minkowski()
    for epoch in range(train["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            views = []
            for view_index in range(2):
                samples = []
                for item in batch:
                    points, colors = augment_points(
                        item["points"], item["colors"], rotation_degrees=train["rotation_degrees"],
                        scale=tuple(train["scale"]), jitter_std_m=train["jitter_std_m"],
                        point_dropout_max=train["point_dropout_max"],
                    )
                    samples.append(sparse_sample(points, colors, pcfg["voxel_size_m"]))
                coordinates, features = me.utils.sparse_collate(
                    [sample["coordinates"] for sample in samples], [sample["features"] for sample in samples]
                )
                sparse = me.SparseTensor(features.float().to(device), coordinates=coordinates.to(device))
                views.append(model(sparse)[0])
            loss = nt_xent_loss(torch.cat(views, dim=0), train["temperature"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        schedule.step()
        print(f"epoch={epoch + 1:03d} loss={total / max(len(loader), 1):.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, args.output)


if __name__ == "__main__":
    main()
