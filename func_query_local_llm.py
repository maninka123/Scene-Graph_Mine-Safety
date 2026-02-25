import importlib.util
import ast
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import torch


def _select_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bf16 when supported; otherwise fp16 on CUDA.
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _safe_token_ids(tokenizer):
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is not None:
        eos_id = int(eos_id)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
        try:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass
    if pad_id is not None:
        pad_id = int(pad_id)
    return eos_id, pad_id


def _balanced_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    start = None
    depth = 0
    in_str = False
    escape = False

    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    return candidates


def _json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.extend(_balanced_json_candidates(text))
    # Prefer later candidates first (LLMs often revise at end).
    return list(reversed(candidates))


def _clean_json_str(s: str) -> str:
    out = s.strip()
    out = re.sub(r",\s*}", "}", out)
    out = re.sub(r",\s*]", "]", out)
    return out


def _normalize_single_response(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        # If model returned a list, convert it into insights.
        if isinstance(data, list):
            return {"insights": data, "hazards": [], "safety_assessment": ""}
        return {"insights": [], "hazards": [], "safety_assessment": str(data)}

    insights = data.get("insights", [])
    if not isinstance(insights, list):
        insights = []

    norm_insights = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id")
        try:
            object_id = int(object_id) if object_id is not None else None
        except Exception:
            object_id = None
        norm_insights.append(
            {
                "object_id": object_id,
                "semantic_label": str(item.get("semantic_label", "")),
                "generated_label": str(item.get("generated_label", "Unknown")),
                "risk_level": str(item.get("risk_level", "")).lower() if item.get("risk_level") is not None else "",
                "reasoning": str(item.get("reasoning", "")),
            }
        )

    hazards = data.get("hazards", [])
    if not isinstance(hazards, list):
        hazards = []

    safety = data.get("safety_assessment", "")
    if safety is None:
        safety = ""

    return {
        "insights": norm_insights,
        "hazards": hazards,
        "safety_assessment": str(safety),
    }


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _map_generated_label(semantic_label: str, volume_m3: float) -> str:
    sem = str(semantic_label or "").lower().strip()
    if sem == "equipment":
        return "Shearer" if volume_m3 >= 8.0 else "Equipment"
    if sem == "conveyor":
        return "Conveyor"
    if sem == "wall":
        return "Wall"
    if sem == "roof":
        return "Roof"
    if sem == "human":
        return "Human"
    if sem == "other":
        return "Debris" if volume_m3 < 1.0 else "Unknown"
    return "Unknown"


def _build_single_fallback(output_dir: str, raw_text: str) -> Dict[str, Any]:
    scene_graph_path = os.path.join(output_dir, "scene_graph.json")
    objects_path = os.path.join(output_dir, "scene_objects_segmented.json")
    scene_graph = _safe_load_json(scene_graph_path) or {}
    nodes = scene_graph.get("nodes", [])
    edges = scene_graph.get("edges", [])

    if not nodes:
        objects = _safe_load_json(objects_path) or []
        if isinstance(objects, list):
            nodes = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                nodes.append(
                    {
                        "id": int(obj.get("id", -1)),
                        "label": str(obj.get("label", "other")),
                        "properties": {
                            "volume_m3": float(obj.get("volume", 0.0) or 0.0),
                            "dimensions": obj.get("dimensions", [0.0, 0.0, 0.0]),
                        },
                    }
                )

    node_by_id: Dict[int, Dict] = {}
    for n in nodes:
        try:
            node_by_id[int(n.get("id", -1))] = n
        except Exception:
            continue

    adjacency: Dict[int, List[Dict[str, Any]]] = {nid: [] for nid in node_by_id.keys()}
    for e in edges:
        try:
            src = int(e.get("source"))
            tgt = int(e.get("target"))
            dist = float(e.get("attributes", {}).get("distance", math.inf))
        except Exception:
            continue
        if src not in adjacency:
            adjacency[src] = []
        if tgt not in adjacency:
            adjacency[tgt] = []
        adjacency[src].append({"id": tgt, "distance": dist})
        adjacency[tgt].append({"id": src, "distance": dist})

    insights: List[Dict[str, Any]] = []
    hazards: List[Dict[str, Any]] = []

    # Human-to-machinery proximity hazards
    machinery_labels = {"equipment", "conveyor"}
    hazard_pairs = []
    for e in edges:
        try:
            src = int(e.get("source"))
            tgt = int(e.get("target"))
            dist = float(e.get("attributes", {}).get("distance", math.inf))
        except Exception:
            continue
        src_label = str(node_by_id.get(src, {}).get("label", "")).lower()
        tgt_label = str(node_by_id.get(tgt, {}).get("label", "")).lower()
        if dist <= 1.5 and (
            (src_label == "human" and tgt_label in machinery_labels)
            or (tgt_label == "human" and src_label in machinery_labels)
        ):
            hazard_pairs.append((src, tgt, dist))

    for src, tgt, dist in hazard_pairs:
        hazards.append(
            {
                "type": "Human-Machinery Proximity",
                "object_ids": [src, tgt],
                "severity": "high",
                "reason": f"Objects {src} and {tgt} are {dist:.2f} m apart (<= 1.5 m).",
            }
        )

    # Machinery clutter hazards
    clutter_candidates = []
    for e in edges:
        try:
            src = int(e.get("source"))
            tgt = int(e.get("target"))
            dist = float(e.get("attributes", {}).get("distance", math.inf))
        except Exception:
            continue
        if dist > 0.8:
            continue
        src_label = str(node_by_id.get(src, {}).get("label", "")).lower()
        tgt_label = str(node_by_id.get(tgt, {}).get("label", "")).lower()
        if (src_label in machinery_labels and tgt_label in {"other", "roof"}) or (
            tgt_label in machinery_labels and src_label in {"other", "roof"}
        ):
            clutter_candidates.append((src, tgt, dist))

    for src, tgt, dist in clutter_candidates[:4]:
        hazards.append(
            {
                "type": "Potential Obstruction Near Machinery",
                "object_ids": [src, tgt],
                "severity": "medium",
                "reason": f"Objects {src} and {tgt} are very close ({dist:.2f} m).",
            }
        )

    for node_id in sorted(node_by_id.keys()):
        node = node_by_id[node_id]
        semantic = str(node.get("label", "other")).lower()
        props = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
        volume = float(props.get("volume_m3", 0.0) or 0.0)
        dims = props.get("dimensions", [0.0, 0.0, 0.0])
        neighbors = adjacency.get(node_id, [])
        min_neighbor_dist = min((float(n["distance"]) for n in neighbors), default=math.inf)

        risk_level = "low"
        if semantic == "human":
            nearest_machinery = math.inf
            for n in neighbors:
                n_label = str(node_by_id.get(int(n["id"]), {}).get("label", "")).lower()
                if n_label in machinery_labels:
                    nearest_machinery = min(nearest_machinery, float(n["distance"]))
            if nearest_machinery <= 1.5:
                risk_level = "high"
            elif nearest_machinery <= 2.5:
                risk_level = "medium"
        elif semantic in machinery_labels:
            if min_neighbor_dist <= 0.8:
                risk_level = "medium"
        elif semantic == "other":
            if min_neighbor_dist <= 0.8:
                risk_level = "medium"

        reasoning = (
            f"Semantic '{semantic}', volume {volume:.2f} m3, dimensions {dims}. "
            + (f"Nearest object at {min_neighbor_dist:.2f} m." if math.isfinite(min_neighbor_dist) else "No nearby objects.")
        )
        insights.append(
            {
                "object_id": int(node_id),
                "semantic_label": semantic,
                "generated_label": _map_generated_label(semantic, volume),
                "risk_level": risk_level,
                "reasoning": reasoning,
            }
        )

    has_high = any(str(h.get("severity", "")).lower() == "high" for h in hazards)
    has_medium = any(str(h.get("severity", "")).lower() == "medium" for h in hazards)
    if has_high:
        safety = "High-risk proximity detected. Review operations and enforce exclusion zones."
    elif has_medium:
        safety = "Moderate operational risk detected. Maintain clearance around moving/critical assets."
    else:
        safety = "No critical proximity hazards detected from current scene graph."

    # Keep a short note from the raw text, but preserve strict schema fields for UI.
    raw_note = raw_text.strip().splitlines()[0][:180] if raw_text.strip() else ""
    if raw_note:
        safety = f"{safety} Note: {raw_note}"

    return {
        "insights": insights,
        "hazards": hazards,
        "safety_assessment": safety,
        "analysis_source": "deterministic_scene_graph_fallback",
    }


def _load_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    for candidate in _json_candidates(response_text):
        candidate = _clean_json_str(candidate)
        try:
            parsed = json.loads(candidate)
            return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: parse Python-style dict/list outputs.
        try:
            py_candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\btrue\b", "True", py_candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\bfalse\b", "False", py_candidate, flags=re.IGNORECASE)
            parsed_py = ast.literal_eval(py_candidate)
            if isinstance(parsed_py, (dict, list)):
                return parsed_py
        except Exception:
            continue
    return None


def _generate_response_text(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 1200) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos_id, pad_id = _safe_token_ids(tokenizer)
    if pad_id is None:
        pad_id = 0
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = getattr(model, "device", torch.device("cpu"))

    inputs = tokenizer([text], return_tensors="pt").to(model_device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    new_tokens = generated_ids[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def query_local_llm(prompt_file, output_dir, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Runs local LLM for single-frame scene graph reasoning.
    Saves normalized JSON to output_dir/LLM/llm_response_real.json.
    """
    print(f"\n{'='*40}")
    print("   [Function] Local LLM Inference")
    print(f"{'='*40}\n")

    root_llm_dir = os.path.join(os.getcwd(), "LLM")
    model_cache_dir = os.path.join(root_llm_dir, "Model_Cache")
    llm_sub_dir = os.path.join(output_dir, "LLM")
    output_response_file = os.path.join(llm_sub_dir, "llm_response_real.json")
    raw_response_file = os.path.join(llm_sub_dir, "llm_response_real.raw.txt")

    os.makedirs(root_llm_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(llm_sub_dir, exist_ok=True)

    print(f"[STEP 1] Loading Prompt: {prompt_file}")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found: {prompt_file}")
        return None

    print(f"[STEP 2] Loading Model: {model_name}...")
    print(f"         Cache Directory: {model_cache_dir}")
    print("         (First run may download model weights)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        model_kwargs = {
            "torch_dtype": _select_torch_dtype(),
            "cache_dir": model_cache_dir,
        }
        if has_accelerate:
            model_kwargs["device_map"] = "auto"
        else:
            print("[WARN] accelerate not installed; loading on a single device.")

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if has_accelerate:
            run_device = str(getattr(model, "device", "cuda:0"))
        else:
            run_device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(run_device)
        print(f"         Loaded on device: {run_device}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    system_prompt = (
        "You are a mining safety expert.\n"
        "Return ONLY strict JSON (no markdown, no prose).\n"
        "Output must start with '{' and end with '}'.\n"
        "Required top-level keys: insights, hazards, safety_assessment.\n"
        "Use evidence from object geometry, semantic labels, and spatial relationships."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_content}]

    print("[STEP 3] Generating Response...")
    try:
        response_text = _generate_response_text(tokenizer, model, messages, max_new_tokens=1200)
    except RuntimeError as e:
        emsg = str(e).lower()
        is_cuda_assert = "device-side assert" in emsg or "cuda error" in emsg
        if is_cuda_assert:
            print("[WARN] CUDA generation failed. Retrying LLM generation on CPU.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                from transformers import AutoModelForCausalLM

                model_cpu = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    cache_dir=model_cache_dir,
                ).to("cpu")
                response_text = _generate_response_text(tokenizer, model_cpu, messages, max_new_tokens=1200)
            except Exception as cpu_e:
                print(f"[ERROR] CPU retry failed: {cpu_e}")
                return None
        else:
            print(f"[ERROR] LLM generation failed: {e}")
            return None

    with open(raw_response_file, "w", encoding="utf-8") as f:
        f.write(response_text)

    print("\n--- LLM Response (preview) ---")
    print(response_text[:1000] if len(response_text) > 1000 else response_text)
    print("------------------------------\n")

    parsed = _load_json_from_text(response_text)
    if parsed is None:
        print("[STEP 4] Retrying with JSON-only repair prompt...")
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Convert the provided text to strict JSON only.\n"
                    "No markdown, no explanation, no extra keys beyond schema if not needed.\n"
                    "Schema: {\"insights\": [...], \"hazards\": [...], \"safety_assessment\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Convert this text to the schema above:\n\n"
                    f"{response_text}"
                ),
            },
        ]
        try:
            repaired_text = _generate_response_text(tokenizer, model, repair_messages, max_new_tokens=1000)
        except RuntimeError:
            # If GPU context is poisoned, force CPU retry.
            try:
                from transformers import AutoModelForCausalLM

                model_cpu = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    cache_dir=model_cache_dir,
                ).to("cpu")
                repaired_text = _generate_response_text(tokenizer, model_cpu, repair_messages, max_new_tokens=1000)
            except Exception:
                repaired_text = ""
        with open(raw_response_file, "a", encoding="utf-8") as f:
            f.write("\n\n=== REPAIR ATTEMPT ===\n")
            f.write(repaired_text)
        parsed = _load_json_from_text(repaired_text)

    if parsed is None:
        print("[INFO] Model output was non-JSON. Using deterministic structured fallback.")
        normalized = _build_single_fallback(output_dir=output_dir, raw_text=response_text)
        with open(output_response_file, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        return output_response_file

    normalized = _normalize_single_response(parsed)
    with open(output_response_file, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    print(f"[SUCCESS] Saved structured insights to: {output_response_file}")

    print(f"\n{'='*40}")
    print("      LLM ANALYSIS REPORT")
    print(f"{'='*40}")
    print(f"\n[SAFETY ASSESSMENT]:\n{normalized.get('safety_assessment', 'N/A')}\n")
    print("[OBJECT INSIGHTS]:")
    for item in normalized.get("insights", []):
        label = item.get("generated_label", "Unknown")
        obj_id = item.get("object_id", "?")
        reasoning = item.get("reasoning", "No reasoning provided.")
        print(f"  - Object {obj_id}: {label}")
        print(f"    Reasoning: {reasoning}")
    print(f"{'='*40}\n")

    return output_response_file


if __name__ == "__main__":
    query_local_llm("llm_prompt.txt", ".")
