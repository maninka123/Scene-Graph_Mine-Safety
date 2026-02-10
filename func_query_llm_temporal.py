from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import re

def query_llm_temporal(prompt_file, output_dir, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Runs the local LLM for temporal scene analysis.
    Saves the response to output_dir/LLM/.
    """
    print(f"\n{'='*50}")
    print(f"   [Function] Temporal LLM Inference")
    print(f"{'='*50}\n")
    
    # 1. Setup Paths
    root_llm_dir = os.path.join(os.getcwd(), "LLM")
    model_cache_dir = os.path.join(root_llm_dir, "Model_Cache")
    
    llm_sub_dir = os.path.join(output_dir, "LLM")
    output_response_file = os.path.join(llm_sub_dir, "llm_temporal_response.json")
    
    os.makedirs(root_llm_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(llm_sub_dir, exist_ok=True)
    
    # 2. Load Prompt
    print(f"[STEP 1] Loading Temporal Prompt: {prompt_file}")
    try:
        with open(prompt_file, "r") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found: {prompt_file}")
        return None

    # 3. Load Model
    print(f"[STEP 2] Loading Model: {model_name}...")
    print(f"         Cache Directory: {model_cache_dir}")
    print(f"         (First run will download ~14GB)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # Fix: use 'dtype' instead of 'torch_dtype'
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            cache_dir=model_cache_dir
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    # 4. Format Prompt for Temporal Analysis
    system_prompt = """You are a mining safety expert analyzing temporal 3D scene data.
Analyze the tracked objects and their movements to understand what is happening in the mine.
Output ONLY valid JSON with the exact structure requested.
Do NOT include any explanations or text outside the JSON object."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 5. Generate
    print(f"[STEP 3] Generating Temporal Analysis (Running on GPU)...")
    import time
    start_time = time.time()
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_token_count = len(inputs.input_ids[0])
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1500,  # More tokens for temporal analysis
            do_sample=False,      # Deterministic (Greedy Search)
            # Removed invalid flags for greedy search (temperature, top_p, top_k)
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    new_tokens = generated_ids[0][len(inputs.input_ids[0]):]
    output_token_count = len(new_tokens)
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Performance Stats
    tokens_per_sec = output_token_count / inference_time if inference_time > 0 else 0
    perf_stats = (
        f"Input Tokens:  {input_token_count}\n"
        f"Output Tokens: {output_token_count}\n"
        f"Total Time:    {inference_time:.2f} sec\n"
        f"Speed:         {tokens_per_sec:.2f} tokens/sec"
    )
    
    print("\n--- LLM Performance ---")
    print(perf_stats)
    print("-----------------------\n")
    
    print("\n--- LLM Response ---")
    print(response_text[:1200] if len(response_text) > 1200 else response_text)
    print("--------------------\n")

    # Save Raw Response
    raw_response_file = os.path.join(llm_sub_dir, "llm_raw_output.txt")
    with open(raw_response_file, "w", encoding="utf-8") as f:
        f.write(f"=== PERFORMANCE METRICS ===\n{perf_stats}\n\n=== RAW RESPONSE ===\n{response_text}")
    print(f"[SUCCESS] Saved raw LLM output to: {raw_response_file}")

    # 6. JSON Extraction & Repair
    def repair_json(json_str):
        json_str = re.sub(r"}\s*{", "}, {", json_str)
        json_str = re.sub(r"\"\s+\"", "\", \"", json_str)
        return json_str

    json_str = None
    
    code_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if code_blocks:
        json_str = code_blocks[-1]
    else:
        potential_jsons = re.findall(r"(\{.*\})", response_text, re.DOTALL)
        if potential_jsons:
            json_str = potential_jsons[-1]
    
    if json_str:
        json_str = re.sub(r",\s*\}", "}", json_str)
        json_str = re.sub(r",\s*\]", "]", json_str)

    data = None
    if json_str:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print("[WARN] Standard JSON parse failed. Attempting fuzzy repair...")
            try:
                json_repaired = repair_json(json_str)
                data = json.loads(json_repaired)
                print("[INFO] Fuzzy repair successful!")
            except json.JSONDecodeError as e:
                print(f"[ERROR] Fuzzy repair failed: {e}")

    if data:
        with open(output_response_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SUCCESS] Saved temporal insights to: {output_response_file}")
        
        # 7. Pretty Print Temporal Results & Save Report
        report_lines = []
        report_lines.append(f"{'='*50}")
        report_lines.append(f"      TEMPORAL LLM ANALYSIS REPORT")
        report_lines.append(f"{'='*50}")
        
        # Add Performance Metrics to Report
        report_lines.append(f"\n[PERFORMANCE METRICS]:")
        report_lines.append(f"  Input Tokens:  {input_token_count}")
        report_lines.append(f"  Output Tokens: {output_token_count}")
        report_lines.append(f"  Inference Time: {inference_time:.2f} sec")
        report_lines.append(f"  Generation Speed: {tokens_per_sec:.2f} tokens/sec")
        
        # Movement Analysis
        if "movement_analysis" in data:
            report_lines.append(f"\n[MOVEMENT ANALYSIS]:")
            report_lines.append(f"  {data['movement_analysis']}")
        
        # Safety Assessment
        if "temporal_safety_assessment" in data:
            report_lines.append(f"\n[TEMPORAL SAFETY ASSESSMENT]:")
            report_lines.append(f"  {data['temporal_safety_assessment']}")
        
        # Object Identifications
        if "object_identifications" in data:
            report_lines.append(f"\n[OBJECT IDENTIFICATIONS]:")
            for item in data.get("object_identifications", []):
                track_id = item.get('track_id', '?')
                identified = item.get('identified_as', 'Unknown')
                confidence = item.get('confidence', 'N/A')
                reasoning = item.get('reasoning', '')
                report_lines.append(f"  • Track {track_id}: {identified} (Confidence: {confidence})")
                if reasoning:
                    report_lines.append(f"    Reason: {reasoning}")
        
        report_lines.append(f"\n{'='*50}\n")
        
        # Print to Console
        full_report = "\n".join(report_lines)
        print(full_report)
        
        # Save to File
        report_file = os.path.join(llm_sub_dir, "llm_analysis_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"[SUCCESS] Saved human-readable report to: {report_file}")
        
        return output_response_file
    else:
        print("[WARNING] No valid JSON found. Saving raw text.")
        raw_file = output_response_file.replace(".json", ".raw.txt")
        with open(raw_file, "w") as f:
            f.write(response_text)
        return None


if __name__ == "__main__":
    query_llm_temporal("llm_temporal_prompt.txt", ".")
