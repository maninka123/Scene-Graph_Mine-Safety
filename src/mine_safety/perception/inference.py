from __future__ import annotations

from pathlib import Path

import torch

from mine_safety.perception.anomaly import anomaly_nodes_from_probabilities
from mine_safety.perception.instances import extract_instances
from mine_safety.perception.io import read_point_cloud, voxelize
from mine_safety.perception.network import SemanticMinkUNet, require_minkowski


class PointCloudPerceiver:
    """Nine-class paper model inference with entropy anomaly extraction."""

    def __init__(self, config: dict, checkpoint: str | Path, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SemanticMinkUNet(
            in_channels=config["perception"]["input_features"],
            width=config["perception"]["backbone_width"],
            feature_dim=config["perception"]["feature_dim"],
            num_classes=len(config["perception"]["classes"]),
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        state = state.get("model_state_dict", state.get("model", state)) if isinstance(state, dict) else state
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, path: str | Path) -> dict:
        me = require_minkowski()
        points, colors = read_point_cloud(path)
        coordinates, features, inverse = voxelize(
            points, colors, self.config["perception"]["voxel_size_m"]
        )
        batched_coordinates = me.utils.batched_coordinates([coordinates])
        sparse = me.SparseTensor(
            coordinates=batched_coordinates.to(self.device),
            features=torch.from_numpy(features).to(self.device),
        )
        logits = self.model(sparse).F
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        labels = probabilities.argmax(axis=1)
        voxel_points = features[:, 3:6]
        instance_config = self.config["perception"]["instances"]
        nodes = extract_instances(
            voxel_points,
            labels,
            probabilities,
            self.config["perception"]["classes"],
            epsilon_m=instance_config["dbscan_epsilon_m"],
            minimum_voxels=instance_config["minimum_voxels"],
        )
        anomaly = self.config["anomaly"]
        nodes.extend(anomaly_nodes_from_probabilities(
            voxel_points, probabilities,
            entropy_threshold=anomaly["entropy_threshold"], minimum_voxels=anomaly["minimum_voxels"],
            epsilon_m=anomaly["dbscan_epsilon_m"], merge_boundary_m=anomaly["merge_boundary_m"],
            merge_centroid_m=anomaly["merge_centroid_m"],
            merge_orientation_degrees=anomaly["merge_orientation_degrees"],
            voxel_size_m=self.config["perception"]["voxel_size_m"],
            closing_iterations=anomaly["closing_iterations"],
        ))
        return {
            "points": points,
            "colors": colors,
            "voxel_points": voxel_points,
            "voxel_labels": labels,
            "voxel_probabilities": probabilities,
            "point_labels": labels[inverse],
            "nodes": nodes,
            "mean_intensity": float(colors.mean() * 255.0),
        }
