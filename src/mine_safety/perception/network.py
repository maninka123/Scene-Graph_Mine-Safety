from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import MinkowskiEngine as ME
except ImportError:  # pragma: no cover - optional CUDA dependency
    ME = None


def require_minkowski() -> Any:
    if ME is None:
        raise ImportError(
            "MinkowskiEngine is required for point-cloud training/inference. "
            "Install a build matching your PyTorch and CUDA versions."
        )
    return ME


class ConvBlock(nn.Module):
    """Two 3x3x3 sparse convolutions, each followed by BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        me = require_minkowski()
        self.block = nn.Sequential(
            me.MinkowskiConvolution(in_channels, out_channels, 3, stride=stride, dimension=3),
            me.MinkowskiBatchNorm(out_channels),
            me.MinkowskiReLU(inplace=True),
            me.MinkowskiConvolution(out_channels, out_channels, 3, stride=1, dimension=3),
            me.MinkowskiBatchNorm(out_channels),
            me.MinkowskiReLU(inplace=True),
        )

    def forward(self, value):
        return self.block(value)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        me = require_minkowski()
        self.up = nn.Sequential(
            me.MinkowskiConvolutionTranspose(in_channels, out_channels, 2, stride=2, dimension=3),
            me.MinkowskiBatchNorm(out_channels),
            me.MinkowskiReLU(inplace=True),
        )
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value, skip):
        me = require_minkowski()
        return self.refine(me.cat(self.up(value), skip))


class MinkUNetBackbone(nn.Module):
    """Paper architecture: 6-D input, 32/64/128/256 hierarchy, 96-D output."""

    def __init__(self, in_channels: int = 6, width: int = 32, feature_dim: int = 96) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.stem = ConvBlock(in_channels, c1)
        self.enc1 = ConvBlock(c1, c1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.enc4 = ConvBlock(c3, c4, stride=2)
        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.output = require_minkowski().MinkowskiConvolution(c1, feature_dim, 1, dimension=3)

    def forward(self, value):
        stem = self.stem(value)
        e1 = self.enc1(stem)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self.output(self.dec1(self.dec2(self.dec3(e4, e3), e2), e1))


class ContrastiveMinkUNet(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 96, projection_dim: int = 128) -> None:
        super().__init__()
        me = require_minkowski()
        self.backbone = MinkUNetBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.pool = me.MinkowskiGlobalAvgPooling()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, value):
        features = self.backbone(value)
        embedding = F.normalize(self.head(self.pool(features).F), dim=1)
        return embedding, features.F


class SemanticMinkUNet(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 96, num_classes: int = 9) -> None:
        super().__init__()
        me = require_minkowski()
        self.backbone = MinkUNetBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.head = me.MinkowskiConvolution(feature_dim, num_classes, 1, dimension=3)

    def forward(self, value):
        return self.head(self.backbone(value))


def nt_xent_loss(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric NT-Xent for [view1 batch, view2 batch] L2-normalised embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] % 2:
        raise ValueError("Expected an even [2B, D] embedding tensor")
    batch = embeddings.shape[0] // 2
    logits = embeddings @ embeddings.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * batch, device=embeddings.device)
    targets = (targets + batch) % (2 * batch)
    return F.cross_entropy(logits, targets)
