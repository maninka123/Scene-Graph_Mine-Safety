from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def _require_word_limit(value: str, maximum: int, field_name: str) -> str:
    if len(value.split()) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} words")
    return value


class ContextualRiskCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    object_ids: list[str]
    evidence: str
    severity: Literal["low", "medium", "high"]
    temporal_pattern: Literal["current_frame", "developing_over_window", "recurring"]

    @field_validator("condition")
    @classmethod
    def condition_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 8, "condition")

    @field_validator("evidence")
    @classmethod
    def evidence_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 30, "evidence")


class ContextualReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hazard_detected: bool
    risk_conditions: list[ContextualRiskCondition] = Field(default_factory=list, max_length=3)
    explanation: str

    @model_validator(mode="after")
    def consistent_hazard_state(self) -> ContextualReasoningResult:
        if self.hazard_detected != bool(self.risk_conditions):
            raise ValueError("hazard_detected must match whether risk_conditions is non-empty")
        return self

    @field_validator("explanation")
    @classmethod
    def explanation_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 40, "explanation")


class LongitudinalRiskCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    object_ids: list[str]
    memory_ids: list[str]
    current_evidence: str
    historical_evidence: str
    severity: Literal["low", "medium", "high"]
    temporal_pattern: Literal["recurring", "escalating"]

    @field_validator("condition")
    @classmethod
    def condition_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 8, "condition")

    @field_validator("current_evidence")
    @classmethod
    def current_evidence_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 30, "current_evidence")

    @field_validator("historical_evidence")
    @classmethod
    def historical_evidence_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 35, "historical_evidence")


class LongitudinalReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hazard_detected: bool
    risk_conditions: list[LongitudinalRiskCondition] = Field(default_factory=list, max_length=3)
    explanation: str

    @model_validator(mode="after")
    def consistent_hazard_state(self) -> LongitudinalReasoningResult:
        if self.hazard_detected != bool(self.risk_conditions):
            raise ValueError("hazard_detected must match whether risk_conditions is non-empty")
        return self

    @field_validator("explanation")
    @classmethod
    def explanation_word_limit(cls, value: str) -> str:
        return _require_word_limit(value, 40, "explanation")


ReasoningResult: TypeAlias = ContextualReasoningResult | LongitudinalReasoningResult
