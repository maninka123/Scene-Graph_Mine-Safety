import importlib.util
import ast
import json
import os
import re
import time
import math
from typing import Any, Dict, List, Optional

import torch


def _select_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
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
    return list(reversed(candidates))


def _clean_json_str(s: str) -> str:
    out = s.strip()
    out = re.sub(r",\s*}", "}", out)
    out = re.sub(r",\s*]", "]", out)
    return out


def _normalize_temporal_response(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "object_identifications": [],
            "movement_analysis": str(data),
            "temporal_hazards": [],
            "temporal_safety_assessment": "",
        }

    ids = data.get("object_identifications", [])
    if not isinstance(ids, list):
        ids = []

    normalized_ids = []
    for item in ids:
        if not isinstance(item, dict):
            continue
        track_id = item.get("track_id")
        try:
            track_id = int(track_id) if track_id is not None else None
        except Exception:
            track_id = None
        normalized_ids.append(
            {
                "track_id": track_id,
                "primary_semantic_label": str(item.get("primary_semantic_label", "")),
                "identified_as": str(item.get("identified_as", "Unknown")),
                "confidence": str(item.get("confidence", "medium")).lower(),
                "reasoning": str(item.get("reasoning", "")),
            }
        )

    hazards = data.get("temporal_hazards", data.get("hazards", []))
    if not isinstance(hazards, list):
        hazards = []

    return {
        "object_identifications": normalized_ids,
        "movement_analysis": str(data.get("movement_analysis", "")),
        "temporal_hazards": hazards,
        "temporal_safety_assessment": str(data.get("temporal_safety_assessment", data.get("safety_assessment", ""))),
    }


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _map_temporal_role(semantic_label: str, avg_volume: float) -> str:
    sem = str(semantic_label or "").lower().strip()
    if sem == "equipment":
        return "Shearer" if avg_volume >= 8.0 else "Equipment"
    if sem == "conveyor":
        return "Conveyor"
    if sem == "human":
        return "Human"
    if sem == "wall":
        return "Wall"
    if sem == "roof":
        return "Roof"
    if sem == "other":
        return "Debris" if avg_volume < 1.0 else "Unknown"
    return "Unknown"


def _build_temporal_fallback(output_dir: str, raw_text: str) -> Dict[str, Any]:
    graph_path = os.path.join(output_dir, "temporal_scene_graph.json")
    temporal_graph = _safe_load_json(graph_path) or {}
    tracked = temporal_graph.get("tracked_objects", []) if isinstance(temporal_graph, dict) else []
    movement_events = temporal_graph.get("movement_events", []) if isinstance(temporal_graph, dict) else []
    summary = temporal_graph.get("summary", {}) if isinstance(temporal_graph, dict) else {}

    object_identifications: List[Dict[str, Any]] = []
    for t in tracked:
        if not isinstance(t, dict):
            continue
        track_id = int(t.get("track_id", -1))
        sem = str(t.get("primary_label", "other")).lower()
        avg_volume = float(t.get("avg_volume", 0.0) or 0.0)
        appearances = int(t.get("num_appearances", 0) or 0)
        displacement = float(t.get("total_displacement", 0.0) or 0.0)
        movement_state = str(t.get("movement_state", "stationary"))

        if appearances >= 4:
            confidence = "high"
        elif appearances >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        reasoning = (
            f"Primary semantic label '{sem}', avg volume {avg_volume:.2f} m3, "
            f"movement state {movement_state}, displacement {displacement:.2f} m."
        )
        object_identifications.append(
            {
                "track_id": track_id,
                "primary_semantic_label": sem,
                "identified_as": _map_temporal_role(sem, avg_volume),
                "confidence": confidence,
                "reasoning": reasoning,
            }
        )

    temporal_hazards: List[Dict[str, Any]] = []
    for ev in movement_events:
        if not isinstance(ev, dict):
            continue
        track_id = int(ev.get("track_id", -1))
        label = str(ev.get("label", "other")).lower()
        state = str(ev.get("state", ""))
        disp = float(ev.get("displacement", 0.0) or 0.0)
        speed = ev.get("speed_m_per_s", None)
        if label == "equipment" and ("fast" in state or disp >= 1.0):
            temporal_hazards.append(
                {
                    "type": "Fast Moving Heavy Equipment",
                    "track_ids": [track_id],
                    "severity": "high" if ("fast" in state or disp >= 2.0) else "medium",
                    "reason": f"Track {track_id} ({label}) moved {disp:.2f} m"
                    + (f" at {float(speed):.2f} m/s." if speed is not None else "."),
                }
            )
        elif label in {"conveyor", "other"} and disp >= 1.0:
            temporal_hazards.append(
                {
                    "type": "Significant Object Movement",
                    "track_ids": [track_id],
                    "severity": "medium",
                    "reason": f"Track {track_id} moved {disp:.2f} m ({state}).",
                }
            )

    moving_count = int(summary.get("moving_objects", len(movement_events)) or 0)
    total_tracks = int(summary.get("total_tracks", len(tracked)) or 0)
    stationary_count = int(summary.get("stationary_objects", max(total_tracks - moving_count, 0)) or 0)
    movement_analysis = (
        f"Tracked {total_tracks} objects over {int(summary.get('total_frames', 0) or 0)} frames: "
        f"{moving_count} moving, {stationary_count} stationary."
    )
    if movement_events:
        top_event = max(
            (ev for ev in movement_events if isinstance(ev, dict)),
            key=lambda x: float(x.get("displacement", 0.0) or 0.0),
            default=None,
        )
        if top_event is not None:
            movement_analysis += (
                f" Largest displacement: track {int(top_event.get('track_id', -1))} "
                f"at {float(top_event.get('displacement', 0.0) or 0.0):.2f} m."
            )

    has_high = any(str(h.get("severity", "")).lower() == "high" for h in temporal_hazards)
    has_medium = any(str(h.get("severity", "")).lower() == "medium" for h in temporal_hazards)
    if has_high:
        safety = "High dynamic risk detected due to fast heavy equipment movement. Enforce exclusion zones."
    elif has_medium:
        safety = "Moderate dynamic risk detected. Monitor moving assets and keep operational clearances."
    else:
        safety = "No critical dynamic hazards detected in the analyzed temporal window."

    raw_note = raw_text.strip().splitlines()[0][:180] if raw_text.strip() else ""
    if raw_note:
        safety = f"{safety} Note: {raw_note}"

    return {
        "object_identifications": object_identifications,
        "movement_analysis": movement_analysis,
        "temporal_hazards": temporal_hazards,
        "temporal_safety_assessment": safety,
        "analysis_source": "deterministic_temporal_graph_fallback",
    }


def _load_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    for candidate in _json_candidates(response_text):
        candidate = _clean_json_str(candidate)
        try:
            return json.loads(candidate)
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


def _generate_response_text(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 1700) -> str:
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


def query_llm_temporal(prompt_file, output_dir, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Runs local LLM for temporal scene analysis.
    Saves normalized JSON to output_dir/LLM/llm_temporal_response.json.
    """
    print(f"\n{'='*50}")
    print("   [Function] Temporal LLM Inference")
    print(f"{'='*50}\n")

    root_llm_dir = os.path.join(os.getcwd(), "LLM")
    model_cache_dir = os.path.join(root_llm_dir, "Model_Cache")
    llm_sub_dir = os.path.join(output_dir, "LLM")
    output_response_file = os.path.join(llm_sub_dir, "llm_temporal_response.json")
    raw_response_file = os.path.join(llm_sub_dir, "llm_raw_output.txt")

    os.makedirs(root_llm_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(llm_sub_dir, exist_ok=True)

    print(f"[STEP 1] Loading Temporal Prompt: {prompt_file}")
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
        "You are a mining safety expert analyzing temporal 3D scene data.\n"
        "Return ONLY strict JSON (no markdown, no prose).\n"
        "Output must start with '{' and end with '}'.\n"
        "Required top-level keys: object_identifications, movement_analysis, temporal_hazards, temporal_safety_assessment.\n"
        "Use semantic labels, geometry, trajectories, and movement dynamics."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_content}]

    print("[STEP 3] Generating Temporal Analysis...")
    start_time = time.time()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_token_count = len(inputs.input_ids[0])
    try:
        response_text = _generate_response_text(tokenizer, model, messages, max_new_tokens=1700)
    except RuntimeError as e:
        emsg = str(e).lower()
        is_cuda_assert = "device-side assert" in emsg or "cuda error" in emsg
        if is_cuda_assert:
            print("[WARN] CUDA generation failed. Retrying temporal LLM generation on CPU.")
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
                response_text = _generate_response_text(tokenizer, model_cpu, messages, max_new_tokens=1700)
            except Exception as cpu_e:
                print(f"[ERROR] CPU retry failed: {cpu_e}")
                return None
        else:
            print(f"[ERROR] LLM generation failed: {e}")
            return None
    end_time = time.time()
    output_token_count = len(tokenizer.encode(response_text, add_special_tokens=False))

    inference_time = max(end_time - start_time, 1e-9)
    tokens_per_sec = output_token_count / inference_time

    perf_stats = (
        f"Input Tokens:  {input_token_count}\n"
        f"Output Tokens: {output_token_count}\n"
        f"Total Time:    {inference_time:.2f} sec\n"
        f"Speed:         {tokens_per_sec:.2f} tokens/sec"
    )
    print("\n--- LLM Performance ---")
    print(perf_stats)
    print("-----------------------\n")

    with open(raw_response_file, "w", encoding="utf-8") as f:
        f.write(f"=== PERFORMANCE METRICS ===\n{perf_stats}\n\n=== RAW RESPONSE ===\n{response_text}")
    print(f"[SUCCESS] Saved raw LLM output to: {raw_response_file}")

    print("\n--- LLM Response (preview) ---")
    print(response_text[:1200] if len(response_text) > 1200 else response_text)
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
                    "Schema: {\"object_identifications\": [...], \"movement_analysis\": \"...\", "
                    "\"temporal_hazards\": [...], \"temporal_safety_assessment\": \"...\"}"
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
            repaired_text = _generate_response_text(tokenizer, model, repair_messages, max_new_tokens=1200)
        except RuntimeError:
            try:
                from transformers import AutoModelForCausalLM

                model_cpu = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    cache_dir=model_cache_dir,
                ).to("cpu")
                repaired_text = _generate_response_text(tokenizer, model_cpu, repair_messages, max_new_tokens=1200)
            except Exception:
                repaired_text = ""
        with open(raw_response_file, "a", encoding="utf-8") as f:
            f.write("\n\n=== REPAIR ATTEMPT ===\n")
            f.write(repaired_text)
        parsed = _load_json_from_text(repaired_text)

    if parsed is None:
        print("[INFO] Model output was non-JSON. Using deterministic structured fallback.")
        normalized = _build_temporal_fallback(output_dir=output_dir, raw_text=response_text)
        with open(output_response_file, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        return output_response_file

    normalized = _normalize_temporal_response(parsed)
    with open(output_response_file, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    print(f"[SUCCESS] Saved temporal insights to: {output_response_file}")

    report_lines = []
    report_lines.append(f"{'='*50}")
    report_lines.append("      TEMPORAL LLM ANALYSIS REPORT")
    report_lines.append(f"{'='*50}")
    report_lines.append("\n[PERFORMANCE METRICS]:")
    report_lines.append(f"  Input Tokens:  {input_token_count}")
    report_lines.append(f"  Output Tokens: {output_token_count}")
    report_lines.append(f"  Inference Time: {inference_time:.2f} sec")
    report_lines.append(f"  Generation Speed: {tokens_per_sec:.2f} tokens/sec")

    report_lines.append("\n[MOVEMENT ANALYSIS]:")
    report_lines.append(f"  {normalized.get('movement_analysis', '')}")

    report_lines.append("\n[TEMPORAL SAFETY ASSESSMENT]:")
    report_lines.append(f"  {normalized.get('temporal_safety_assessment', '')}")

    report_lines.append("\n[OBJECT IDENTIFICATIONS]:")
    for item in normalized.get("object_identifications", []):
        track_id = item.get("track_id", "?")
        identified = item.get("identified_as", "Unknown")
        confidence = item.get("confidence", "N/A")
        reasoning = item.get("reasoning", "")
        report_lines.append(f"  - Track {track_id}: {identified} (Confidence: {confidence})")
        if reasoning:
            report_lines.append(f"    Reason: {reasoning}")
    report_lines.append(f"\n{'='*50}\n")

    full_report = "\n".join(report_lines)
    print(full_report)
    report_file = os.path.join(llm_sub_dir, "llm_analysis_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"[SUCCESS] Saved human-readable report to: {report_file}")

    return output_response_file


if __name__ == "__main__":
    query_llm_temporal("llm_temporal_prompt.txt", ".")
