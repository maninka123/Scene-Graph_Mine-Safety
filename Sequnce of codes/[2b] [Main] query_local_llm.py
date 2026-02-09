from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

# ==========================================
#              INPUT VARIABLES
# ==========================================
PROMPT_FILE = "llm_prompt.txt"
# Store model and output in 'LLM' directory as requested
LLM_DIR = "LLM"
MODEL_CACHE_DIR = os.path.join(LLM_DIR, "DeepSeek_Model")
OUTPUT_RESPONSE_FILE = os.path.join(LLM_DIR, "llm_response_real.json")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 

def run_local_llm():
    print(f"\n{'='*40}")
    print(f"   Local LLM Inference (DeepSeek-1.5B)")
    print(f"{'='*40}\n")
    
    # Ensure LLM directory exists
    os.makedirs(LLM_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # 1. Load Prompt
    print(f"[STEP 1] Loading Prompt: {PROMPT_FILE}")
    try:
        with open(PROMPT_FILE, "r") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found. Run [2] build_graph.py first.")
        return

    # 2. Load Model
    print(f"[STEP 2] Loading Model: {MODEL_NAME}...")
    print(f"         Storing weights in: {MODEL_CACHE_DIR}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 3. Format Prompt
    # DeepSeek/Qwen instruct format
    messages = [
        {"role": "system", "content": "You are an expert mining engineer and safety inspector. Output JSON only."},
        {"role": "user", "content": prompt_content + "\n\nProvide your analysis in JSON format with keys: 'insights' (list of objects with 'object_id', 'generated_label', 'reasoning') and 'safety_assessment' (string)."}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 4. Generate
    print(f"[STEP 3] Generating Response (Running on GPU)...")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # Low temp for factual/structured output
            do_sample=False
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- LLM Raw Output ---")
    print(response_text)
    print("----------------------\n")

    # 5. Save (Parsing naive JSON)
    # We attempt to extract JSON content
    try:
        # Simple heuristic to find JSON start/end
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        
        if start != -1 and end != -1:
            json_str = response_text[start:end]
            data = json.loads(json_str)
            
            with open(OUTPUT_RESPONSE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[SUCCESS] Saved structured response to: {OUTPUT_RESPONSE_FILE}")
            print(f"         Now execute: python \"[3] [Visualisation] visualize_scene_graph.py\"")
        else:
            print("[WARNING] Could not parse JSON from output. Showing raw text instead.")
            print(response_text)
    except Exception as e:
        print(f"[ERROR] JSON Parsing failed: {e}")
        print("Raw Output:")
        print(response_text)

if __name__ == "__main__":
    run_local_llm()
