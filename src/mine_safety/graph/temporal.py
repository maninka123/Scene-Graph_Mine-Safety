from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
from math import sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment

from mine_safety.schemas import SceneGraph


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class TemporalGraphTracker:
    """Hungarian object association over the paper's rolling 10-second window."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        label_penalty: float = 1000.0,
        max_displacement_m: float = 1.0,
        stationary_m: float = 0.2,
        slow_m: float = 0.5,
    ) -> None:
        self.window_seconds = window_seconds
        self.label_penalty = label_penalty
        self.max_displacement_m = max_displacement_m
        self.stationary_m = stationary_m
        self.slow_m = slow_m
        self.frames: deque[SceneGraph] = deque()
        self._next_track = 1

    def _new_track_id(self) -> str:
        value = f"track-{self._next_track:04d}"
        self._next_track += 1
        return value

    def update(self, graph: SceneGraph) -> SceneGraph:
        current = deepcopy(graph)
        if not self.frames:
            for node in current.nodes:
                node.track_id = self._new_track_id()
                node.movement_state = "unknown"
        else:
            previous = self.frames[-1]
            dt = max((current.timestamp - previous.timestamp).total_seconds(), 1e-6)
            costs = np.full((len(previous.nodes), len(current.nodes)), np.inf, dtype=float)
            for i, old in enumerate(previous.nodes):
                for j, new in enumerate(current.nodes):
                    displacement = _distance(old.centroid, new.centroid)
                    if displacement <= self.max_displacement_m:
                        costs[i, j] = displacement + (self.label_penalty if old.label != new.label else 0.0)
            matched_new: set[int] = set()
            if costs.size and np.isfinite(costs).any():
                safe = np.where(np.isfinite(costs), costs, 1e12)
                rows, cols = linear_sum_assignment(safe)
                for row, col in zip(rows, cols):
                    if not np.isfinite(costs[row, col]) or costs[row, col] >= self.label_penalty:
                        continue
                    old, new = previous.nodes[row], current.nodes[col]
                    delta = tuple(new.centroid[k] - old.centroid[k] for k in range(3))
                    displacement = _distance(old.centroid, new.centroid)
                    new.track_id = old.track_id or self._new_track_id()
                    new.velocity = tuple(value / dt for value in delta)
                    new.movement_state = (
                        "stationary" if displacement < self.stationary_m
                        else "slow" if displacement <= self.slow_m
                        else "fast"
                    )
                    matched_new.add(col)
            for idx, node in enumerate(current.nodes):
                if idx not in matched_new:
                    node.track_id = self._new_track_id()
                    node.movement_state = "unknown"

        self.frames.append(current)
        cutoff = current.timestamp.timestamp() - self.window_seconds
        while self.frames and self.frames[0].timestamp.timestamp() < cutoff:
            self.frames.popleft()
        return current

    def snapshot(self) -> dict:
        tracks: dict[str, list[dict]] = {}
        for frame in self.frames:
            for node in frame.nodes:
                if node.track_id:
                    tracks.setdefault(node.track_id, []).append(
                        {
                            "timestamp": frame.timestamp.isoformat(),
                            "node_id": node.id,
                            "label": node.label,
                            "centroid": node.centroid,
                            "velocity": node.velocity,
                            "movement_state": node.movement_state,
                            "entropy": node.entropy,
                            "is_anomaly": node.is_anomaly,
                        }
                    )
        return {
            "window_seconds": self.window_seconds,
            "frame_count": len(self.frames),
            "frames": [frame.model_dump(mode="json") for frame in self.frames],
            "tracks": tracks,
        }
