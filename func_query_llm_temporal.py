import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch


def _select_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


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
    import re

    candidates: List[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.extend(_balanced_json_candidates(text))
    return list(reversed(candidates))


def _clean_json_str(s: str) -> str:
    import re

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


def _load_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    for candidate in _json_candidates(response_text):
        candidate = _clean_json_str(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


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
        "Return ONLY strict JSON with the requested schema.\n"
        "Use semantic labels, geometry, trajectories, and movement dynamics."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("[STEP 3] Generating Temporal Analysis...")
    start_time = time.time()
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_token_count = len(inputs.input_ids[0])
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1700,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()

    new_tokens = generated_ids[0][len(inputs.input_ids[0]) :]
    output_token_count = len(new_tokens)
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

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
        print("[WARNING] Could not parse valid JSON. Saved raw response only.")
        return None

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
