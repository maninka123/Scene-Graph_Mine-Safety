from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

def query_local_llm(prompt_file, output_dir, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Runs the local LLM using the provided prompt file.
    Saves the response to output_dir.
    """
    import re
    
    print(f"\n{'='*40}")
    print(f"   [Function] Local LLM Inference")
    print(f"{'='*40}\n")
    
    # 1. Setup Paths
    # Model Cache: Centralized in root/LLM/Model_Cache (NOT in Results)
    root_llm_dir = os.path.join(os.getcwd(), "LLM") 
    model_cache_dir = os.path.join(root_llm_dir, "Model_Cache")
    
    # Output: Saved in the specific Results folder
    llm_sub_dir = os.path.join(output_dir, "LLM")
    output_response_file = os.path.join(llm_sub_dir, "llm_response_real.json")
    
    os.makedirs(root_llm_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    os.makedirs(llm_sub_dir, exist_ok=True)
    
    # 2. Load Prompt
    print(f"[STEP 1] Loading Prompt: {prompt_file}")
    try:
        with open(prompt_file, "r") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found: {prompt_file}")
        return None

    # Clean up root prompt if it exists
    root_prompt = "llm_prompt.txt"
    if os.path.exists(root_prompt) and os.path.abspath(root_prompt) != os.path.abspath(prompt_file):
        try:
            os.remove(root_prompt)
        except:
            pass

    # 3. Load Model
    print(f"[STEP 2] Loading Model: {model_name}...")
    print(f"         Cache Directory: {model_cache_dir}")
    print(f"         (First run will download ~14GB)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,  # Qwen2.5 works best with bfloat16
            device_map="auto",
            cache_dir=model_cache_dir
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

    # 4. Format Prompt (Optimized for Qwen2.5 Instruct)
    system_prompt = """You are a mining safety expert. Analyze the 3D scene data and output ONLY valid JSON.
Do NOT include any explanations, thinking, or text outside the JSON object.
Output format: {"insights": [{"object_id": int, "generated_label": string, "reasoning": string}, ...], "safety_assessment": string}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 5. Generate
    print(f"[STEP 3] Generating Response (Running on GPU)...")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # Qwen is concise, doesn't need as much
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decode only the new tokens
    new_tokens = generated_ids[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("\n--- LLM Response ---")
    print(response_text[:1000] if len(response_text) > 1000 else response_text)
    print("--------------------\n")

    # 6. Robust JSON Extraction & Repair
    def repair_json(json_str):
        # Fix missing commas between objects in a list: } { -> }, {
        json_str = re.sub(r"}\s*{", "}, {", json_str)
        # Fix missing commas between fields: " " -> ", " (with newline check)
        # This is risky but helps with common LLM errors like: "key": "val" "key2": "val"
        # We look for: "value"\s+"key":
        json_str = re.sub(r"\"\s+\"", "\", \"", json_str)
        return json_str

    json_str = None
    
    # Try finding the LAST code block first
    code_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if code_blocks:
        json_str = code_blocks[-1]
    else:
        # Try finding the LAST valid JSON-like object {...}
        potential_jsons = re.findall(r"(\{.*\})", response_text, re.DOTALL)
        if potential_jsons:
            json_str = potential_jsons[-1]
            
    # Clean up common JSON errors (trailing commas)
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
        print(f"[SUCCESS] Saved structured insights to: {output_response_file}")
        
        # 7. Pretty Print Results (User Request)
        print(f"\n{'='*40}")
        print(f"      LLM ANALYSIS REPORT")
        print(f"{'='*40}")
        print(f"\n[SAFETY ASSESSMENT]:\n{data.get('safety_assessment', 'N/A')}\n")
        print(f"[OBJECT INSIGHTS]:")
        for item in data.get("insights", []):
            label = item.get('generated_label', 'Unknown')
            obj_id = item.get('object_id', '?')
            reasoning = item.get('reasoning', 'No reasoning provided.')
            print(f"  • Object {obj_id}: {label}")
            print(f"    Reasoning: {reasoning}")
        print(f"{'='*40}\n")
        
        return output_response_file
    else:
        print("[WARNING] No valid JSON found. Saving raw text.")
        with open(output_response_file.replace(".json", ".raw.txt"), "w") as f:
            f.write(response_text)
        return None

if __name__ == "__main__":
    # Test
    query_local_llm("llm_prompt.txt", ".")
