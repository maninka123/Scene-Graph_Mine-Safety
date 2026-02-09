import json
import numpy as np
import math

# ==========================================
#              INPUT VARIABLES
# ==========================================
INPUT_OBJECTS_FILE = "scene_objects_clustered.json"  # UPDATED to use clustered data
OUTPUT_GRAPH_FILE = "scene_graph.json"
OUTPUT_PROMPT_FILE = "llm_prompt.txt"

# Graph Construction Parameters
DISTANCE_THRESHOLD = 2.5  # Increased slightly for clusters
ALIGNMENT_THRESHOLD = 2.0 
# ==========================================

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

# Helper for volume if missing
def bbox_volume(dims):
    return dims[0] * dims[1] * dims[2]

def build_graph():
    print(f"\n{'='*40}")
    print(f"      Scene Graph Builder (Cluster-Based)")
    print(f"{'='*40}\n")

    # 1. Load Data
    print(f"[STEP 1] Loading Objects from: {INPUT_OBJECTS_FILE}")
    try:
        with open(INPUT_OBJECTS_FILE, "r") as f:
            objects = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {INPUT_OBJECTS_FILE}")
        return
        
    if not objects:
        print("[WARNING] No objects found! Run segment_clustering.py first.")
        return
    else:
        print(f"         Loaded {len(objects)} objects.")

    # 2. Build Nodes
    print(f"[STEP 2] Constructing Graph Nodes...")
    nodes = []
    for obj in objects:
        # Include physical properties for the LLM to reason about
        props = {
            "position": obj["centroid"],
            "volume_m3": obj.get("volume", 0) if "volume" in obj else bbox_volume(obj.get("dimensions", [0,0,0])),
            "dimensions": obj.get("dimensions", [0,0,0])
        }
        
        nodes.append({
            "id": obj["id"],
            "label": obj["label"], # e.g. "large_machine..."
            "properties": props
        })
        
    # 3. Build Edges (Relationships)
    print(f"[STEP 3] analyzing Spatial Relationships...")
    edges = []
    relationship_count = 0
    
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_a = objects[i]
            obj_b = objects[j]
            
            dist = calculate_distance(obj_a["centroid"], obj_b["centroid"])
            
            # Relation: Proximity
            if dist < DISTANCE_THRESHOLD:
                edges.append({
                    "source": obj_a["id"],
                    "target": obj_b["id"],
                    "relation": "near",
                    "attributes": {"distance": round(dist, 2)}
                })
                relationship_count += 1
                 
    print(f"         Identified {relationship_count} relationships.")

    scene_graph = {
        "nodes": nodes,
        "edges": edges
    }
    
    # 4. Save Graph
    with open(OUTPUT_GRAPH_FILE, "w") as f:
        json.dump(scene_graph, f, indent=2)
    print(f"\n[SUCCESS] Scene Graph saved to: {OUTPUT_GRAPH_FILE}")
    
    # 5. Generate LLM Prompt
    print(f"\n[STEP 4] Generating LLM Prompt...")
    
    # Enhanced Prompt for reasoning with generic labels
    prompt = "I have a 3D scene graph of an underground longwall mine face, derived from unlabelled point cloud clusters.\n"
    prompt += "Your task is to identify the objects and assess safety based on their dimensions and arrangement.\n\n"
    
    prompt += "Context:\n"
    prompt += "- 'large_machine_or_structure' is likely a Hydraulic Support (Chock) or the Shearer.\n"
    prompt += "- Supports are usually arranged in a row.\n"
    prompt += "- The Shearer is a single large mobile machine, often near the supports but aligned with the face.\n\n"
    
    prompt += "Detected Objects:\n"
    for node in nodes:
        pos = [round(x, 1) for x in node['properties']['position']]
        dims = [round(x, 1) for x in node['properties']['dimensions']]
        vol = round(node['properties'].get('volume_m3', 0), 1)
        
        prompt += f"- Object {node['id']} ({node['label']}):\n"
        prompt += f"    Position: {pos}\n"
        prompt += f"    Dimensions (LxWxH): {dims}\n"
        prompt += f"    Volume: {vol} m3\n"
        
    prompt += "\nRelationships:\n"
    for edge in edges:
        prompt += f"- Object {edge['source']} is {edge['relation']} Object {edge['target']} (dist: {edge['attributes'].get('distance', 'N/A')}m)\n"
        
    prompt += "\nQuestions for the LLM:\n"
    prompt += "1. Based on the dimensions and arrangement, which object is likely the Shearer?\n"
    prompt += "2. Identify the Hydraulic Supports (Chocks) based on their repetitive arrangement.\n"
    prompt += "3. Is there a safe distance (>1.0m) between the Shearer and the Supports?\n"
    
    # Save prompt to file for user
    with open(OUTPUT_PROMPT_FILE, "w") as f:
        f.write(prompt)
    print(f"[SUCCESS] Prompt saved to: {OUTPUT_PROMPT_FILE}")
    
    print(f"\n{'='*40}")
    print(" PREVIEW OF PROMPT ")
    print(f"{'='*40}")
    print(prompt.strip()[:1000])

if __name__ == "__main__":
    build_graph()
