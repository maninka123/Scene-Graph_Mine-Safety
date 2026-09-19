from __future__ import annotations

import json
import re
from typing import Any, Callable

from mine_safety.reasoning.prompts import CONTEXTUAL_SYSTEM_PROMPT, LONGITUDINAL_SYSTEM_PROMPT
from mine_safety.schemas import ReasoningResult, SafetyAlert, SceneGraph


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Model output did not contain a valid JSON object")


def validate_grounding(
    result: ReasoningResult,
    graph: SceneGraph,
    memory_ids: set[str] | None = None,
) -> None:
    object_ids = {node.id for node in graph.nodes}
    memories = memory_ids or set()
    for hazard in result.hazards:
        unknown_objects = set(hazard.object_ids) - object_ids
        if unknown_objects:
            raise ValueError(f"LLM referenced unknown object ids: {sorted(unknown_objects)}")
        unknown_memories = set(hazard.memory_ids) - memories
        if unknown_memories:
            raise ValueError(f"LLM referenced unknown memory ids: {sorted(unknown_memories)}")


def serialize_reasoning_input(graph: SceneGraph, temporal: dict, alerts: list[SafetyAlert]) -> str:
    payload = {
        "current_scene_graph": graph.model_dump(mode="json"),
        "temporal_graph": temporal,
        "deterministic_alerts": [alert.model_dump(mode="json") for alert in alerts],
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class LocalQwenReasoner:
    """On-device Qwen adapter with schema and graph-grounding validation."""

    def __init__(self, config: dict, generator: Callable[[list[dict[str, str]]], str] | None = None) -> None:
        self.config = config
        self.generator = generator
        self._pipeline = None

    def _generate(self, messages: list[dict[str, str]]) -> str:
        if self.generator is not None:
            return self.generator(messages)
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            self._pipeline = pipeline(
                "text-generation",
                model=self.config["contextual_model"],
                device_map="auto",
                model_kwargs={"torch_dtype": "auto"},
            )
        output = self._pipeline(
            messages,
            max_new_tokens=self.config["max_new_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            do_sample=True,
        )
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)

    def reason(
        self,
        graph: SceneGraph,
        temporal: dict,
        alerts: list[SafetyAlert],
        memories: list[dict] | None = None,
    ) -> ReasoningResult:
        memory_ids = {str(item["memory_id"]) for item in memories or []}
        system = LONGITUDINAL_SYSTEM_PROMPT if memories else CONTEXTUAL_SYSTEM_PROMPT
        user_payload = serialize_reasoning_input(graph, temporal, alerts)
        if memories:
            user_payload += "\nRETRIEVED_MEMORIES=" + json.dumps(memories, separators=(",", ":"))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_payload}]
        attempts = 1 + int(self.config.get("regeneration_attempts", 1))
        error: Exception | None = None
        for _ in range(attempts):
            try:
                result = ReasoningResult.model_validate(_extract_json(self._generate(messages)))
                validate_grounding(result, graph, memory_ids)
                return result
            except Exception as exc:  # validation triggers the paper's one regeneration attempt
                error = exc
                messages.append({
                    "role": "user",
                    "content": f"Previous output failed validation ({exc}). Return corrected strict JSON only.",
                })
        raise RuntimeError(f"Contextual output rejected after {attempts} attempt(s): {error}")
