from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _class_color_hex(class_colors: Dict[str, List[int]], class_name: str) -> str:
    rgb = class_colors.get(class_name, [200, 200, 200])
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def build_label_color_map(class_names: List[str], class_colors: Dict[str, List[int]]) -> Dict[int, str]:
    return {idx: _class_color_hex(class_colors, name) for idx, name in enumerate(class_names)}


def sample_points(points: np.ndarray, labels: np.ndarray, max_points: int = 30000) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, labels
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], labels[idx]


def scatter3d_semantic(
    points: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    class_colors: Dict[str, List[int]],
    title: str,
    max_points: int = 30000,
):
    pts, lbs = sample_points(points, labels, max_points=max_points)
    rows = []
    for xyz, lb in zip(pts, lbs):
        class_name = class_names[int(lb)] if int(lb) >= 0 and int(lb) < len(class_names) else "unlabeled"
        rows.append({"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]), "class": class_name})
    df = pd.DataFrame(rows)
    color_map = {name: _class_color_hex(class_colors, name) for name in class_names}
    color_map["unlabeled"] = "rgb(30,30,30)"
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="class",
        color_discrete_map=color_map,
        title=title,
        opacity=0.9,
    )
    fig.update_traces(marker={"size": 2})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        legend_title_text="Class",
        legend={"itemsizing": "constant"},
    )
    return fig


def bar_class_counts(class_counts: Dict[str, int], title: str):
    labels = list(class_counts.keys())
    values = [int(class_counts[k]) for k in labels]
    fig = px.bar(
        x=labels,
        y=values,
        labels={"x": "Class", "y": "Points"},
        title=title,
        text=values,
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig


def line_frame_metrics(frame_metrics: pd.DataFrame, x_col: str = "frame_index"):
    fig = go.Figure()
    if "inference_seconds" in frame_metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=frame_metrics[x_col],
                y=frame_metrics["inference_seconds"],
                mode="lines+markers",
                name="Inference Time (s)",
            )
        )
    if "num_objects" in frame_metrics.columns:
        fig.add_trace(
            go.Scatter(
                x=frame_metrics[x_col],
                y=frame_metrics["num_objects"],
                mode="lines+markers",
                name="Objects",
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Temporal Inference Metrics",
        xaxis={"title": "Frame Index"},
        yaxis={"title": "Inference Time (s)"},
        yaxis2={"title": "Objects", "overlaying": "y", "side": "right"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h"},
    )
    return fig


def bar_movement_states(temporal_graph: dict):
    tracked = temporal_graph.get("tracked_objects", [])
    counts: Dict[str, int] = {"stationary": 0, "moving_slow": 0, "moving_fast": 0}
    for item in tracked:
        state = str(item.get("movement_state", "stationary"))
        counts[state] = counts.get(state, 0) + 1
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        labels={"x": "Movement State", "y": "Tracked Objects"},
        title="Movement State Distribution",
        text=list(counts.values()),
    )
    return fig


def safe_frame_metrics_df(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["frame_index", "frame_name", "inference_seconds", "num_points", "num_objects"])
    return pd.DataFrame(rows)
