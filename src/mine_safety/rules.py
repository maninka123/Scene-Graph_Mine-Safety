from __future__ import annotations

from collections import deque
from math import acos, degrees, pi, sqrt

import numpy as np

from mine_safety.schemas import SafetyAlert, SceneGraph, SceneNode


def _norm(vector) -> float:
    return sqrt(sum(float(value) ** 2 for value in vector))


def _bbox_clearance(a: SceneNode, b: SceneNode) -> float:
    ac, bc = np.asarray(a.centroid), np.asarray(b.centroid)
    ah, bh = np.asarray(a.bbox_dimensions) / 2.0, np.asarray(b.bbox_dimensions) / 2.0
    axis_gap = np.maximum(np.abs(ac - bc) - ah - bh, 0.0)
    return float(np.linalg.norm(axis_gap))


class SafetyRuleEngine:
    """Five authoritative deterministic checks from Section 2.5.1."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.personnel = {item.lower() for item in config["personnel_labels"]}
        self.equipment = {item.lower() for item in config["equipment_labels"]}
        self.intensity_history: deque[tuple[float, float]] = deque()

    def evaluate(self, graph: SceneGraph) -> list[SafetyAlert]:
        for edge in graph.edges:
            edge.safety_flags.clear()
        alerts: list[SafetyAlert] = []
        people = [node for node in graph.nodes if node.label.lower() in self.personnel]
        machines = [node for node in graph.nodes if node.label.lower() in self.equipment and node.active]
        for machine in machines:
            nearby_people = []
            for person in people:
                delta = np.asarray(person.centroid) - np.asarray(machine.centroid)
                centroid_distance = float(np.linalg.norm(delta))
                half_diagonal_sum = _norm(person.bbox_dimensions) / 2 + _norm(machine.bbox_dimensions) / 2
                clearance = centroid_distance
                if centroid_distance <= self.config["safe_clearance_m"] + half_diagonal_sum:
                    clearance = _bbox_clearance(person, machine)
                if centroid_distance <= self.config["safe_clearance_m"] or clearance <= self.config["safe_clearance_m"]:
                    alerts.append(SafetyAlert(
                        rule="proximity_violation", severity="critical",
                        object_ids=[person.id, machine.id],
                        message="Personnel is inside the configured equipment exclusion clearance.",
                        evidence={"centroid_distance_m": centroid_distance, "bbox_clearance_m": clearance},
                    ))
                if centroid_distance <= self.config["congestion_radius_m"]:
                    nearby_people.append(person)
                relative_velocity = np.asarray(machine.velocity) - np.asarray(person.velocity)
                direction = delta / max(centroid_distance, 1e-12)
                closing_speed = float(np.dot(relative_velocity, direction))
                if closing_speed > 0:
                    ttc = (centroid_distance - self.config["safe_clearance_m"]) / closing_speed
                    if 0 <= ttc <= self.config["ttc_threshold_seconds"]:
                        alerts.append(SafetyAlert(
                            rule="ttc_warning", severity="critical",
                            object_ids=[person.id, machine.id], message="Predicted time-to-collision is below threshold.",
                            evidence={"ttc_seconds": ttc, "closing_speed_mps": closing_speed},
                        ))
                speed = _norm(machine.velocity)
                if speed > 1e-6 and centroid_distance <= self.config["blind_spot_distance_m"]:
                    heading = np.asarray(machine.orientation, dtype=float)
                    if np.dot(heading, machine.velocity) < 0:
                        heading = -heading
                    cosine = float(np.dot(delta, heading)) / max(centroid_distance * np.linalg.norm(heading), 1e-12)
                    angle = degrees(acos(np.clip(cosine, -1.0, 1.0)))
                    if angle > self.config["blind_spot_angle_degrees"]:
                        alerts.append(SafetyAlert(
                            rule="blind_spot", severity="high", object_ids=[person.id, machine.id],
                            message="Personnel occupies the moving equipment rear-sector warning zone.",
                            evidence={"angle_degrees": angle, "distance_m": centroid_distance},
                        ))
            density = len(nearby_people) / (pi * self.config["congestion_radius_m"] ** 2)
            if density > self.config["congestion_density_per_m2"]:
                alerts.append(SafetyAlert(
                    rule="congestion", severity="high",
                    object_ids=[machine.id, *[node.id for node in nearby_people]],
                    message="Personnel density around active equipment exceeds the configured limit.",
                    evidence={"personnel_count": len(nearby_people), "density_per_m2": density},
                ))
        if graph.mean_intensity is not None:
            now = graph.timestamp.timestamp()
            self.intensity_history.append((now, graph.mean_intensity))
            cutoff = now - self.config["visibility_window_seconds"]
            while self.intensity_history and self.intensity_history[0][0] < cutoff:
                self.intensity_history.popleft()
            rolling = sum(value for _, value in self.intensity_history) / len(self.intensity_history)
            observed_window = self.intensity_history[-1][0] - self.intensity_history[0][0]
            if (
                observed_window >= self.config["visibility_window_seconds"]
                and rolling < self.config["visibility_intensity_threshold"]
            ):
                alerts.append(SafetyAlert(
                    rule="low_visibility", severity="high", object_ids=[],
                    message="Rolling mean scene intensity indicates severely degraded visibility.",
                    evidence={"rolling_mean_intensity": rolling},
                ))
        graph.metadata["deterministic_safety_flags"] = [
            alert.model_dump(mode="json") for alert in alerts
        ]
        for alert in alerts:
            involved = set(alert.object_ids)
            for edge in graph.edges:
                if edge.source in involved and edge.target in involved and alert.rule not in edge.safety_flags:
                    edge.safety_flags.append(alert.rule)
        return alerts
