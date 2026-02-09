import json
import numpy as np
import math
import os

# Helper for volume if missing
def bbox_volume(dims):
    return dims[0] * dims[1] * dims[2]

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def build_scene_graph(objects_file, output_dir, dist_threshold=2.5):
    """
    Constructs a scene graph from clustered objects.
    Returns the path to the generated LLM prompt file.
    """
    print(f"\n{'='*40}")
    print(f"   [Function] Scene Graph Construction")
    print(f"{'='*40}\n")

    # Output Paths
    output_graph_file = os.path.join(output_dir, "scene_graph.json")
    output_prompt_file = os.path.join(output_dir, "llm_prompt.txt")

    # 1. Load Data
    print(f"[STEP 1] Loading Objects from: {objects_file}")
    try:
        with open(objects_file, "r") as f:
            objects = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {objects_file}")
        return None
        
    if not objects:
        print("[WARNING] No objects found!")
        return None
    else:
        print(f"         Loaded {len(objects)} objects.")

    # 2. Build Nodes
    print(f"[STEP 2] Constructing Graph Nodes...")
    nodes = []
    for obj in objects:
        props = {
            "position": obj["centroid"],
            "volume_m3": obj.get("volume", 0) if "volume" in obj else bbox_volume(obj.get("dimensions", [0,0,0])),
            "dimensions": obj.get("dimensions", [0,0,0])
        }
        
        nodes.append({
            "id": obj["id"],
            "label": obj["label"], 
            "properties": props
        })
        
    # 3. Build Edges
    print(f"[STEP 3] analyzing Spatial Relationships...")
    edges = []
    relationship_count = 0
    
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_a = objects[i]
            obj_b = objects[j]
            
            dist = calculate_distance(obj_a["centroid"], obj_b["centroid"])
            
            if dist < dist_threshold:
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
    with open(output_graph_file, "w") as f:
        json.dump(scene_graph, f, indent=2)
    print(f"\n[SUCCESS] Scene Graph saved to: {output_graph_file}")
    
    # 5. Generate LLM Prompt
    print(f"\n[STEP 4] Generating LLM Prompt...")
    
    # Analyze Basic Stats for Intro
    num_objects = len(nodes)
    large_objs = sum(1 for n in nodes if n['properties'].get('volume_m3', 0) > 5.0)
    
    prompt = f"I have a 3D scene graph of an underground longwall mine face, derived from {num_objects} point cloud clusters.\n"
    prompt += "Your task is to identify the objects (Shearer, Chocks) and assess safety based on their dimensions and arrangement.\n\n"
    
    prompt += "### Scene Summary\n"
    prompt += f"- Total Objects Detected: {num_objects}\n"
    prompt += f"- Large Structures (>5m3): {large_objs} (Likely major machinery)\n\n"
    
    prompt += "### Mining Context (ground truth knowledge)\n"
    prompt += "1. **Hydraulic Supports (Chocks)**: Large rectangular structures (~2m x 1.5m x 2m+), usually arranged in a consistent row with uniform spacing.\n"
    prompt += "2. **Shearer**: A very large mobile machine (often >8m3), longer than a chock, positioned along the face.\n"
    prompt += "3. **Safety Rule**: Humans or obstacles should not be within 1.0m of the Shearer during operation.\n\n"
    
    prompt += "### Detected Objects (Data from 3D Scanner)\n"
    for node in nodes:
        pos = [round(x, 1) for x in node['properties']['position']]
        dims = [round(x, 1) for x in node['properties']['dimensions']]
        vol = round(node['properties'].get('volume_m3', 0), 1)
        
        prompt += f"- **Object {node['id']}** ({node['label']}):\n"
        prompt += f"    - Position (XYZ): {pos}\n"
        prompt += f"    - Dimensions (LxWxH): {dims}\n"
        prompt += f"    - Volume: {vol} m3\n"
        
    prompt += "\n### Spatial Relationships\n"
    if not edges:
        prompt += "No close proximity relationships detected.\n"
    else:
        for edge in edges:
            prompt += f"- Object {edge['source']} is NEAR Object {edge['target']} (Distance: {edge['attributes'].get('distance', 'N/A')}m)\n"
        
    prompt += "\n### Analysis Request\n"
    prompt += "Return a JSON object with:\n"
    prompt += "1. `insights`: A list of objects. For each detected object, provide:\n"
    prompt += "   - `object_id`: (int)\n"
    prompt += "   - `generated_label`: (string, e.g., 'Shearer', 'Hydraulic Chock', 'Debris')\n"
    prompt += "   - `reasoning`: (string, explain why based on dimensions/position)\n"
    prompt += "2. `safety_assessment`: (string) A one-sentence safety check focusing on the Shearer's proximity to other objects.\n"
    
    with open(output_prompt_file, "w") as f:
        f.write(prompt)
    print(f"[SUCCESS] Prompt saved to: {output_prompt_file}")
    
    return output_prompt_file

if __name__ == "__main__":
    build_scene_graph("scene_objects_clustered.json", ".")
