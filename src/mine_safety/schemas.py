from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Vector3 = tuple[float, float, float]


class SceneNode(BaseModel):
    id: str
    label: str
    centroid: Vector3
    bbox_dimensions: Vector3 = (0.5, 0.5, 1.0)
    orientation: Vector3 = (1.0, 0.0, 0.0)
    volume_m3: float = 0.0
    voxel_count: int = 1
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entropy: float = Field(default=0.0, ge=0.0)
    is_anomaly: bool = False
    active: bool = True
    velocity: Vector3 = (0.0, 0.0, 0.0)
    movement_state: Literal["stationary", "slow", "fast", "unknown"] = "unknown"
    track_id: str | None = None


class SceneEdge(BaseModel):
    source: str
    target: str
    distance_m: float = Field(ge=0.0)
    relation: str = "near"
    safety_flags: list[str] = Field(default_factory=list)


class SceneGraph(BaseModel):
    graph_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: list[SceneNode]
    edges: list[SceneEdge] = Field(default_factory=list)
    mean_intensity: float | None = Field(default=None, ge=0.0, le=255.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def unique_node_ids(cls, nodes: list[SceneNode]) -> list[SceneNode]:
        ids = [node.id for node in nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        return nodes


class SafetyAlert(BaseModel):
    rule: str
    severity: Literal["low", "medium", "high", "critical"]
    object_ids: list[str]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class RiskCondition(BaseModel):
    condition: str
    object_ids: list[str]
    severity: Literal["low", "medium", "high"]
    status: Literal["observed", "developing", "unresolved", "recurring", "escalating"]
    explanation: str
    current_evidence: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    historical_evidence: list[str] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    hazards: list[RiskCondition] = Field(default_factory=list, max_length=3)
    summary: str
    no_additional_hazard: bool = False

    @model_validator(mode="after")
    def consistent_empty_state(self) -> "ReasoningResult":
        if self.no_additional_hazard and self.hazards:
            raise ValueError("no_additional_hazard cannot be true when hazards are present")
        return self
