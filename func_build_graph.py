import json
import numpy as np
import math
import os

# Helper for volume if missing
def bbox_volume(dims):
    return dims[0] * dims[1] * dims[2]

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def _safe_float_list(values, n=3):
    out = [0.0] * n
    if not isinstance(values, (list, tuple)):
        return out
    for i in range(min(n, len(values))):
        try:
            out[i] = float(values[i])
        except Exception:
            out[i] = 0.0
    return out


def _object_node(obj):
    dims = _safe_float_list(obj.get("dimensions", [0, 0, 0]), n=3)
    centroid = _safe_float_list(obj.get("centroid", [0, 0, 0]), n=3)
    volume = obj.get("volume", None)
    if volume is None:
        volume = bbox_volume(dims)
    try:
        volume = float(volume)
    except Exception:
        volume = bbox_volume(dims)
    return {
        "id": int(obj.get("id", -1)),
        "label": str(obj.get("label", "unknown")),
        "properties": {
            "position": centroid,
            "volume_m3": volume,
            "dimensions": dims,
            "point_count": int(obj.get("point_count", 0)),
            "confidence": float(obj.get("confidence", 1.0)),
        },
    }

def build_scene_graph(objects_file, output_dir, dist_threshold=2.5):
    """
    Constructs a scene graph from segmented objects.
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
        nodes.append(_object_node(obj))
        
    # 3. Build Edges
    print(f"[STEP 3] analyzing Spatial Relationships...")
    edges = []
    relationship_count = 0
    
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_a = objects[i]
            obj_b = objects[j]
            
            centroid_a = _safe_float_list(obj_a.get("centroid", [0, 0, 0]), n=3)
            centroid_b = _safe_float_list(obj_b.get("centroid", [0, 0, 0]), n=3)
            dist = calculate_distance(centroid_a, centroid_b)
            
            if dist < dist_threshold:
                edges.append({
                    "source": int(obj_a.get("id", i)),
                    "target": int(obj_b.get("id", j)),
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
    semantic_to_ids = {}
    for node in nodes:
        semantic_to_ids.setdefault(node["label"], []).append(node["id"])
    
    prompt = f"I have a 3D scene graph of an underground longwall mine face, derived from {num_objects} segmented point cloud objects.\n"
    prompt += "You are given semantic labels from segmentation and geometric features from the scene graph.\n"
    prompt += "Infer likely mining-role labels and provide a safety assessment with concise, evidence-based reasoning.\n\n"
    
    prompt += "### Scene Summary\n"
    prompt += f"- Total Objects Detected: {num_objects}\n"
    prompt += f"- Large Structures (>5m3): {large_objs} (Likely major machinery)\n\n"

    semantic_counts = {}
    for node in nodes:
        semantic_counts[node["label"]] = semantic_counts.get(node["label"], 0) + 1
    prompt += "### Semantic Class Counts\n"
    for cls_name, cls_count in sorted(semantic_counts.items(), key=lambda x: x[0]):
        prompt += f"- {cls_name}: {cls_count}\n"
    prompt += "\n"

    # Deterministic risk context to anchor reasoning.
    equipment_ids = set(semantic_to_ids.get("equipment", []))
    conveyor_ids = set(semantic_to_ids.get("conveyor", []))
    human_ids = set(semantic_to_ids.get("human", []))
    known_moving_ids = equipment_ids.union(conveyor_ids)
    proximity_alerts = []
    for edge in edges:
        a = edge.get("source")
        b = edge.get("target")
        dist = edge.get("attributes", {}).get("distance", None)
        if dist is None:
            continue
        if ((a in human_ids and b in known_moving_ids) or (b in human_ids and a in known_moving_ids)) and float(dist) <= 1.5:
            proximity_alerts.append((a, b, float(dist)))

    prompt += "### Deterministic Pre-LLM Risk Cues\n"
    if not proximity_alerts:
        prompt += "- No human-to-equipment/conveyor near-contact <= 1.5 m detected from graph edges.\n"
    else:
        for a, b, d in sorted(proximity_alerts, key=lambda x: x[2]):
            prompt += f"- Potential proximity hazard: Object {a} and Object {b} at {d:.2f} m\n"
    prompt += "\n"
    
    prompt += "### Mining Context (ground truth knowledge)\n"
    prompt += "1. **Hydraulic Supports (Chocks)**: Large rectangular structures (~2m x 1.5m x 2m+), usually arranged in a consistent row with uniform spacing.\n"
    prompt += "2. **Shearer**: A very large mobile machine (often >8m3), longer than a chock, positioned along the face.\n"
    prompt += "3. **Safety Rule**: Humans or obstacles should not be within 1.0m of the Shearer during operation.\n"
    prompt += "4. **Segmentation classes**: wall, equipment, human, conveyor, roof, other.\n\n"
    
    prompt += "### Detected Objects (Data from 3D Scanner)\n"
    for node in nodes:
        pos = [round(x, 1) for x in node['properties']['position']]
        dims = [round(x, 1) for x in node['properties']['dimensions']]
        vol = round(node['properties'].get('volume_m3', 0), 1)
        pcount = int(node["properties"].get("point_count", 0))
        conf = float(node["properties"].get("confidence", 1.0))
        
        prompt += f"- **Object {node['id']}** ({node['label']}):\n"
        prompt += f"    - Position (XYZ): {pos}\n"
        prompt += f"    - Dimensions (LxWxH): {dims}\n"
        prompt += f"    - Volume: {vol} m3\n"
        prompt += f"    - Points: {pcount}\n"
        prompt += f"    - Segmentation confidence: {conf:.2f}\n"
        
    prompt += "\n### Spatial Relationships\n"
    if not edges:
        prompt += "No close proximity relationships detected.\n"
    else:
        sorted_edges = sorted(edges, key=lambda e: float(e.get("attributes", {}).get("distance", 1e9)))
        for edge in sorted_edges:
            prompt += f"- Object {edge['source']} is NEAR Object {edge['target']} (Distance: {edge['attributes'].get('distance', 'N/A')}m)\n"
        
    prompt += "\n### Analysis Request\n"
    prompt += "Return ONLY valid JSON (no markdown, no extra text) with:\n"
    prompt += "1. `insights`: A list of objects. For each detected object, provide:\n"
    prompt += "   - `object_id`: (int)\n"
    prompt += "   - `semantic_label`: (string; one of wall/equipment/human/conveyor/roof/other)\n"
    prompt += "   - `generated_label`: (string; e.g., 'Shearer', 'Hydraulic Chock', 'Conveyor', 'Wall', 'Human', 'Debris', 'Unknown')\n"
    prompt += "   - `risk_level`: (string; one of low/medium/high)\n"
    prompt += "   - `reasoning`: (string; reference dimensions/volume/proximity)\n"
    prompt += "2. `hazards`: list of hazards; each item has `type`, `object_ids`, `severity`, `reason`\n"
    prompt += "3. `safety_assessment`: (string) concise scene-level safety assessment.\n"
    
    with open(output_prompt_file, "w") as f:
        f.write(prompt)
    print(f"[SUCCESS] Prompt saved to: {output_prompt_file}")
    
    return output_prompt_file

if __name__ == "__main__":
    build_scene_graph("scene_objects_segmented.json", ".")
