from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


def _parse_header_line(header_lines: Dict[str, str], key: str) -> str:
    if key not in header_lines:
        raise ValueError(f"Missing {key} in PCD header")
    return header_lines[key]


def _decode_packed_rgb(rgb_values: np.ndarray) -> np.ndarray:
    rgb_uint = rgb_values.astype(np.uint32)
    r = (rgb_uint >> 16) & 255
    g = (rgb_uint >> 8) & 255
    b = rgb_uint & 255
    return np.stack([r, g, b], axis=1).astype(np.float32) / 255.0


def read_pcd(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PCD file not found: {p}")

    with p.open("r", encoding="utf-8", errors="ignore") as handle:
        raw_header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PCD format: {p}")
            line = line.strip()
            raw_header.append(line)
            if line.upper().startswith("DATA"):
                break

        header: Dict[str, str] = {}
        for line in raw_header:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].upper()] = parts[1]

        data_type = _parse_header_line(header, "DATA").split()[0].lower()
        if data_type != "ascii":
            pcd = o3d.io.read_point_cloud(str(p))
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.float32)
            if colors.size == 0:
                colors = np.zeros((points.shape[0], 3), dtype=np.float32)
            return points, colors

        fields = _parse_header_line(header, "FIELDS").split()
        data = np.loadtxt(handle, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]

    field_to_idx = {field.lower(): idx for idx, field in enumerate(fields)}
    for axis in ("x", "y", "z"):
        if axis not in field_to_idx:
            raise ValueError(f"PCD missing {axis} field: {p}")

    xyz = np.stack(
        [
            data[:, field_to_idx["x"]],
            data[:, field_to_idx["y"]],
            data[:, field_to_idx["z"]],
        ],
        axis=1,
    ).astype(np.float32)

    if "rgb" in field_to_idx:
        colors = _decode_packed_rgb(data[:, field_to_idx["rgb"]])
    elif all(k in field_to_idx for k in ("r", "g", "b")):
        colors = np.stack(
            [
                data[:, field_to_idx["r"]],
                data[:, field_to_idx["g"]],
                data[:, field_to_idx["b"]],
            ],
            axis=1,
        ).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    return xyz, colors


def write_colored_pcd(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.clip(colors, 0.0, 1.0)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(p), cloud, write_ascii=True)

