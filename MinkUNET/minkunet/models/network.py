from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep thread count bounded on first import if user did not set it.
os.environ.setdefault("OMP_NUM_THREADS", "12")

try:
    import MinkowskiEngine as ME
except ImportError:  # pragma: no cover - runtime check
    ME = None


def ensure_minkowski_available() -> None:
    if ME is None:
        raise ImportError(
            "MinkowskiEngine is not installed. Install a CUDA-compatible build for your PyTorch version "
            "before running MinkUNET scripts."
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        ensure_minkowski_available()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        ensure_minkowski_available()
        self.up = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )
        self.merge = ConvBlock(out_channels + skip_channels, out_channels, stride=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = ME.cat(x, skip)
        return self.merge(x)


class MinkUNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 96, width: int = 32):
        super().__init__()
        ensure_minkowski_available()

        c1 = width
        c2 = width * 2
        c3 = width * 4
        c4 = width * 8

        self.stem = ConvBlock(in_channels, c1, stride=1)
        self.enc1 = ConvBlock(c1, c1, stride=1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.enc4 = ConvBlock(c3, c4, stride=2)

        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

        self.out_proj = ME.MinkowskiConvolution(c1, feature_dim, kernel_size=1, stride=1, dimension=3)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_proj(d1)


class ContrastiveMinkUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 96,
        projection_dim: int = 128,
        width: int = 32,
    ):
        super().__init__()
        ensure_minkowski_available()

        self.backbone = MinkUNetBackbone(in_channels=in_channels, feature_dim=feature_dim, width=width)
        self.pool = ME.MinkowskiGlobalAvgPooling()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse_features = self.backbone(x)
        pooled = self.pool(sparse_features).F
        embeddings = self.projection_head(pooled)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings, sparse_features.F


class SemanticMinkUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 96,
        num_classes: int = 6,
        width: int = 32,
    ):
        super().__init__()
        ensure_minkowski_available()

        self.backbone = MinkUNetBackbone(in_channels=in_channels, feature_dim=feature_dim, width=width)
        self.classifier = ME.MinkowskiConvolution(
            in_channels=feature_dim,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            dimension=3,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def sparse_tensor_from_batch(coords: torch.Tensor, features: torch.Tensor, device: torch.device):
    ensure_minkowski_available()
    return ME.SparseTensor(
        coordinates=coords.to(device),
        features=features.to(device),
    )
