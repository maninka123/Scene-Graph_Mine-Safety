import importlib.util
import json
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


def _load_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    for candidate in _json_candidates(response_text):
        candidate = _clean_json_str(candidate)
        try:
            parsed = json.loads(candidate)
            return parsed
        except json.JSONDecodeError:
            continue
    return None


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
        "Return ONLY strict JSON. Do not include markdown or extra prose.\n"
        "Use evidence from object geometry, semantic labels, and spatial relationships."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("[STEP 3] Generating Response...")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated_ids[0][len(inputs.input_ids[0]) :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    with open(raw_response_file, "w", encoding="utf-8") as f:
        f.write(response_text)

    print("\n--- LLM Response (preview) ---")
    print(response_text[:1000] if len(response_text) > 1000 else response_text)
    print("------------------------------\n")

    parsed = _load_json_from_text(response_text)
    if parsed is None:
        print("[WARNING] Could not parse valid JSON. Saved raw response only.")
        return None

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
