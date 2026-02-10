import json
import numpy as np
import math
import os
from scipy.optimize import linear_sum_assignment

def calculate_distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def match_objects_between_frames(prev_objects, curr_objects, max_distance=1.0):
    """
    Match objects between consecutive frames using Hungarian algorithm.
    Returns mapping: {curr_obj_id: prev_obj_id} or {curr_obj_id: None} for new objects.
    """
    if not prev_objects or not curr_objects:
        return {obj["id"]: None for obj in curr_objects}
    
    n_prev = len(prev_objects)
    n_curr = len(curr_objects)
    
    # Build cost matrix (distance between centroids)
    cost_matrix = np.zeros((n_curr, n_prev))
    for i, curr_obj in enumerate(curr_objects):
        for j, prev_obj in enumerate(prev_objects):
            dist = calculate_distance(curr_obj["centroid"], prev_obj["centroid"])
            cost_matrix[i, j] = dist
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build mapping (only accept matches within max_distance)
    mapping = {}
    for curr_idx, prev_idx in zip(row_ind, col_ind):
        if cost_matrix[curr_idx, prev_idx] <= max_distance:
            mapping[curr_objects[curr_idx]["id"]] = prev_objects[prev_idx]["id"]
        else:
            mapping[curr_objects[curr_idx]["id"]] = None
    
    # Mark unmatched current objects
    for obj in curr_objects:
        if obj["id"] not in mapping:
            mapping[obj["id"]] = None
    
    return mapping

def classify_movement(displacement, threshold_slow=0.2, threshold_fast=0.5):
    """Classify movement based on displacement."""
    if displacement < threshold_slow:
        return "stationary"
    elif displacement < threshold_fast:
        return "moving_slow"
    else:
        return "moving_fast"

def calculate_direction(start_pos, end_pos):
    """Calculate movement direction vector."""
    vec = np.array(end_pos) - np.array(start_pos)
    length = np.linalg.norm(vec)
    if length < 0.001:
        return [0, 0, 0]
    return (vec / length).tolist()

def build_temporal_graph(frame_files, output_dir, dist_threshold=2.5, 
                          max_track_distance=1.0, movement_threshold=0.2):
    """
    Builds a temporal scene graph with object tracking across frames.
    Returns path to the temporal scene graph JSON file.
    """
    print(f"\n{'='*50}")
    print(f"   [Function] Temporal Scene Graph Construction")
    print(f"{'='*50}\n")
    
    # Output paths
    output_graph_file = os.path.join(output_dir, "temporal_scene_graph.json")
    output_prompt_file = os.path.join(output_dir, "llm_temporal_prompt.txt")
    
    # 1. Load all frames
    print(f"[STEP 1] Loading {len(frame_files)} frame files...")
    all_frames = []
    for frame_file in frame_files:
        try:
            with open(frame_file, "r") as f:
                objects = json.load(f)
                all_frames.append(objects)
        except Exception as e:
            print(f"         [WARN] Could not load {frame_file}: {e}")
            all_frames.append([])
    
    if not all_frames or all(len(f) == 0 for f in all_frames):
        print("[ERROR] No valid frame data found.")
        return None
    
    # 2. Track objects across frames
    print(f"[STEP 2] Tracking objects across frames...")
    
    # Global track ID counter
    next_track_id = 0
    
    # Track info: {track_id: {"positions": [(frame, pos), ...], "labels": [...], ...}}
    tracks = {}
    
    # Mapping from (frame_id, local_obj_id) -> track_id
    obj_to_track = {}
    
    prev_objects = []
    prev_track_mapping = {}  # {local_obj_id: track_id}
    
    for frame_idx, frame_objects in enumerate(all_frames):
        if not frame_objects:
            prev_objects = []
            prev_track_mapping = {}
            continue
        
        # Match with previous frame
        matching = match_objects_between_frames(prev_objects, frame_objects, max_track_distance)
        
        curr_track_mapping = {}
        
        for obj in frame_objects:
            obj_id = obj["id"]
            matched_prev_id = matching.get(obj_id)
            
            if matched_prev_id is not None and matched_prev_id in prev_track_mapping:
                # Continue existing track
                track_id = prev_track_mapping[matched_prev_id]
            else:
                # Start new track
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "track_id": track_id,
                    "positions": [],
                    "timestamps": [],
                    "labels": [],
                    "volumes": [],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx
                }
            
            # Update track
            tracks[track_id]["positions"].append(obj["centroid"])
            tracks[track_id]["timestamps"].append(obj.get("timestamp", str(frame_idx)))
            tracks[track_id]["labels"].append(obj.get("label", "unknown"))
            tracks[track_id]["volumes"].append(obj.get("volume", 0))
            tracks[track_id]["last_frame"] = frame_idx
            
            curr_track_mapping[obj_id] = track_id
            obj_to_track[(frame_idx, obj_id)] = track_id
        
        prev_objects = frame_objects
        prev_track_mapping = curr_track_mapping
    
    print(f"         Identified {len(tracks)} unique object tracks.")
    
    # 3. Analyze track movements
    print(f"[STEP 3] Analyzing movement patterns...")
    
    tracked_objects = []
    movement_events = []
    
    for track_id, track in tracks.items():
        positions = track["positions"]
        
        if len(positions) < 2:
            # Single appearance
            displacement = 0.0
            velocity = 0.0
            direction = [0, 0, 0]
        else:
            # Calculate total displacement
            start_pos = positions[0]
            end_pos = positions[-1]
            displacement = calculate_distance(start_pos, end_pos)
            
            # Average velocity (frames are roughly 10 seconds apart based on metadata)
            num_frames = len(positions)
            velocity = displacement / max(num_frames - 1, 1)
            
            direction = calculate_direction(start_pos, end_pos)
        
        movement_state = classify_movement(displacement, movement_threshold)
        
        # Most common label
        label_counts = {}
        for lbl in track["labels"]:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        primary_label = max(label_counts, key=label_counts.get)
        
        # Average volume
        avg_volume = sum(track["volumes"]) / len(track["volumes"]) if track["volumes"] else 0
        
        track_summary = {
            "track_id": track_id,
            "primary_label": primary_label,
            "avg_volume": round(avg_volume, 2),
            "first_frame": track["first_frame"],
            "last_frame": track["last_frame"],
            "num_appearances": len(positions),
            "start_position": positions[0] if positions else None,
            "end_position": positions[-1] if positions else None,
            "total_displacement": round(displacement, 3),
            "avg_velocity": round(velocity, 4),
            "direction": [round(d, 3) for d in direction],
            "movement_state": movement_state,
            "trajectory": positions
        }
        tracked_objects.append(track_summary)
        
        # Log significant movements
        if movement_state != "stationary":
            movement_events.append({
                "track_id": track_id,
                "label": primary_label,
                "displacement": round(displacement, 2),
                "direction": [round(d, 2) for d in direction],
                "state": movement_state
            })
    
    print(f"         Found {len(movement_events)} significant movement events.")
    
    # 4. Build spatial relationships for first and last frames
    print(f"[STEP 4] Building spatial relationships...")
    
    def build_frame_edges(frame_objects, dist_thresh):
        edges = []
        for i in range(len(frame_objects)):
            for j in range(i + 1, len(frame_objects)):
                dist = calculate_distance(
                    frame_objects[i]["centroid"], 
                    frame_objects[j]["centroid"]
                )
                if dist < dist_thresh:
                    edges.append({
                        "source": frame_objects[i]["id"],
                        "target": frame_objects[j]["id"],
                        "distance": round(dist, 2)
                    })
        return edges
    
    first_frame_edges = build_frame_edges(all_frames[0], dist_threshold) if all_frames[0] else []
    last_frame_edges = build_frame_edges(all_frames[-1], dist_threshold) if all_frames[-1] else []
    
    # 5. Compile temporal graph
    temporal_graph = {
        "summary": {
            "total_frames": len(all_frames),
            "total_tracks": len(tracked_objects),
            "moving_objects": len(movement_events),
            "stationary_objects": len(tracked_objects) - len(movement_events)
        },
        "tracked_objects": tracked_objects,
        "movement_events": movement_events,
        "spatial_relationships": {
            "first_frame": first_frame_edges,
            "last_frame": last_frame_edges
        }
    }
    
    with open(output_graph_file, "w") as f:
        json.dump(temporal_graph, f, indent=2)
    print(f"\n[SUCCESS] Temporal graph saved to: {output_graph_file}")
    
    # 6. Generate LLM Prompt
    print(f"[STEP 5] Generating temporal LLM prompt...")
    
    prompt = generate_temporal_prompt(temporal_graph, all_frames)
    
    with open(output_prompt_file, "w") as f:
        f.write(prompt)
    print(f"[SUCCESS] Temporal prompt saved to: {output_prompt_file}")
    
    return output_prompt_file


def generate_temporal_prompt(temporal_graph, all_frames):
    """Generate a prompt for LLM temporal analysis."""
    
    summary = temporal_graph["summary"]
    tracked = temporal_graph["tracked_objects"]
    movements = temporal_graph["movement_events"]
    
    prompt = f"""I have a TEMPORAL 3D scene graph from an underground longwall mine face.
The data spans {summary['total_frames']} consecutive point cloud frames captured approximately 10 seconds apart.

Your task is to:
1. Identify the mining equipment (Shearer, Hydraulic Chocks) based on size and behavior
2. Analyze movement patterns to understand what is happening in the scene
3. Assess safety considering dynamic movements

### Temporal Summary
- Total Frames Analyzed: {summary['total_frames']}
- Unique Objects Tracked: {summary['total_tracks']}
- Objects with Significant Movement: {summary['moving_objects']}
- Stationary Objects: {summary['stationary_objects']}

### Mining Equipment Context
1. **Hydraulic Supports (Chocks)**: Large rectangular (~2m x 1.5m x 2m+), typically STATIONARY in a row.
2. **Shearer**: Very large mobile machine (>8m³), expected to MOVE along the coal face.
3. **Safety Rule**: Monitor Shearer movements and proximity to workers/obstacles.

### Tracked Objects (with Movement Analysis)
"""
    
    for obj in tracked:
        prompt += f"\n**Track {obj['track_id']}** ({obj['primary_label']}):\n"
        prompt += f"  - Volume: {obj['avg_volume']} m³\n"
        prompt += f"  - Appeared in frames: {obj['first_frame']} to {obj['last_frame']} ({obj['num_appearances']} appearances)\n"
        prompt += f"  - Movement: {obj['movement_state'].upper()}\n"
        if obj['movement_state'] != 'stationary':
            prompt += f"  - Displacement: {obj['total_displacement']}m\n"
            prompt += f"  - Direction: {obj['direction']}\n"
    
    if movements:
        prompt += "\n### Significant Movement Events\n"
        for m in movements:
            prompt += f"- Track {m['track_id']} ({m['label']}): Moved {m['displacement']}m ({m['state']})\n"
    
    prompt += """
### Analysis Request
Return a JSON object with:
1. `object_identifications`: List of objects with:
   - `track_id`: (int)
   - `identified_as`: (string: 'Shearer', 'Hydraulic Chock', 'Debris', 'Unknown')
   - `confidence`: (string: 'high', 'medium', 'low')
   - `reasoning`: (string)

2. `movement_analysis`: (string) Description of what is happening in the scene based on movements.

3. `temporal_safety_assessment`: (string) Safety assessment considering the dynamic movements over time.
"""
    
    return prompt


if __name__ == "__main__":
    # Test with dummy data
    test_files = ["Results/01_test/frames/frame_00_objects.json"]
    build_temporal_graph(test_files, "Results/01_test")
