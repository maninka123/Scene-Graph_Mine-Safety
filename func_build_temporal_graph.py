import json
import math
import os
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def _parse_timestamp_any(ts_value):
    if ts_value is None:
        return None
    if isinstance(ts_value, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_value))
        except Exception:
            return None

    ts = str(ts_value).strip()
    if not ts:
        return None

    # ISO format (used in app temporal runner)
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        pass

    # Pattern from filenames: 20260122_153641662
    try:
        if "_" in ts and len(ts.split("_")) == 2:
            d, t = ts.split("_")
            hms = t[:6]
            frac = t[6:]
            base = datetime.strptime(d + hms, "%Y%m%d%H%M%S")
            if frac:
                usec = int(frac[:6].ljust(6, "0"))
                base = base.replace(microsecond=usec)
            return base
    except Exception:
        pass
    return None


def match_objects_between_frames(prev_objects, curr_objects, max_distance=1.0):
    """
    Match objects between consecutive frames using Hungarian algorithm.
    Returns mapping: {curr_obj_id: prev_obj_id} or {curr_obj_id: None} for new objects.
    """
    if not prev_objects or not curr_objects:
        return {obj["id"]: None for obj in curr_objects}

    n_prev = len(prev_objects)
    n_curr = len(curr_objects)

    # Cost matrix combines distance + strong penalty for class mismatch.
    cost_matrix = np.zeros((n_curr, n_prev), dtype=np.float64)
    for i, curr_obj in enumerate(curr_objects):
        for j, prev_obj in enumerate(prev_objects):
            dist = calculate_distance(curr_obj["centroid"], prev_obj["centroid"])
            same_label = str(curr_obj.get("label", "")) == str(prev_obj.get("label", ""))
            cost_matrix[i, j] = dist if same_label else (dist + 1e3)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for curr_idx, prev_idx in zip(row_ind, col_ind):
        dist = calculate_distance(curr_objects[curr_idx]["centroid"], prev_objects[prev_idx]["centroid"])
        if dist <= max_distance:
            mapping[curr_objects[curr_idx]["id"]] = prev_objects[prev_idx]["id"]
        else:
            mapping[curr_objects[curr_idx]["id"]] = None

    for obj in curr_objects:
        if obj["id"] not in mapping:
            mapping[obj["id"]] = None

    return mapping


def classify_movement(displacement, threshold_slow=0.2, threshold_fast=0.5):
    """Classify movement based on displacement."""
    if displacement < threshold_slow:
        return "stationary"
    if displacement < threshold_fast:
        return "moving_slow"
    return "moving_fast"


def calculate_direction(start_pos, end_pos):
    """Calculate movement direction vector."""
    vec = np.array(end_pos) - np.array(start_pos)
    length = np.linalg.norm(vec)
    if length < 1e-3:
        return [0, 0, 0]
    return (vec / length).tolist()


def build_temporal_graph(
    frame_files,
    output_dir,
    dist_threshold=2.5,
    max_track_distance=1.0,
    movement_threshold=0.2,
):
    """
    Builds a temporal scene graph with object tracking across frames.
    Returns path to the temporal prompt file.
    """
    print(f"\n{'='*50}")
    print("   [Function] Temporal Scene Graph Construction")
    print(f"{'='*50}\n")

    output_graph_file = os.path.join(output_dir, "temporal_scene_graph.json")
    output_prompt_file = os.path.join(output_dir, "llm_temporal_prompt.txt")

    print(f"[STEP 1] Loading {len(frame_files)} frame files...")
    all_frames = []
    for frame_file in frame_files:
        try:
            with open(frame_file, "r", encoding="utf-8") as f:
                objects = json.load(f)
                all_frames.append(objects)
        except Exception as e:
            print(f"         [WARN] Could not load {frame_file}: {e}")
            all_frames.append([])

    if not all_frames or all(len(f) == 0 for f in all_frames):
        print("[ERROR] No valid frame data found.")
        return None

    print("[STEP 2] Tracking objects across frames...")
    next_track_id = 0
    tracks = {}

    prev_objects = []
    prev_track_mapping = {}

    for frame_idx, frame_objects in enumerate(all_frames):
        if not frame_objects:
            prev_objects = []
            prev_track_mapping = {}
            continue

        matching = match_objects_between_frames(prev_objects, frame_objects, max_track_distance)
        curr_track_mapping = {}

        for obj in frame_objects:
            obj_id = obj["id"]
            matched_prev_id = matching.get(obj_id)

            if matched_prev_id is not None and matched_prev_id in prev_track_mapping:
                track_id = prev_track_mapping[matched_prev_id]
            else:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "track_id": track_id,
                    "positions": [],
                    "timestamps": [],
                    "labels": [],
                    "volumes": [],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                }

            tracks[track_id]["positions"].append(obj.get("centroid", [0, 0, 0]))
            tracks[track_id]["timestamps"].append(obj.get("timestamp", str(frame_idx)))
            tracks[track_id]["labels"].append(obj.get("label", "unknown"))
            tracks[track_id]["volumes"].append(float(obj.get("volume", 0.0)))
            tracks[track_id]["last_frame"] = frame_idx

            curr_track_mapping[obj_id] = track_id

        prev_objects = frame_objects
        prev_track_mapping = curr_track_mapping

    print(f"         Identified {len(tracks)} unique object tracks.")

    print("[STEP 3] Analyzing movement patterns...")
    tracked_objects = []
    movement_events = []

    threshold_fast = max(float(movement_threshold) * 2.5, float(movement_threshold) + 0.1)
    frame_deltas_for_summary = []

    for track_id, track in tracks.items():
        positions = track["positions"]
        timestamps = track["timestamps"]

        if len(positions) < 2:
            displacement = 0.0
            velocity_frame = 0.0
            direction = [0, 0, 0]
        else:
            start_pos = positions[0]
            end_pos = positions[-1]
            displacement = calculate_distance(start_pos, end_pos)
            velocity_frame = displacement / max(len(positions) - 1, 1)
            direction = calculate_direction(start_pos, end_pos)

        # Time-aware speed if timestamps are parseable
        parsed_ts = [_parse_timestamp_any(t) for t in timestamps]
        speed_mps = None
        if len(parsed_ts) >= 2 and all(t is not None for t in parsed_ts):
            elapsed = (parsed_ts[-1] - parsed_ts[0]).total_seconds()
            if elapsed > 1e-6:
                speed_mps = displacement / elapsed
            for i in range(1, len(parsed_ts)):
                dt = (parsed_ts[i] - parsed_ts[i - 1]).total_seconds()
                if dt > 0:
                    frame_deltas_for_summary.append(dt)

        movement_state = classify_movement(displacement, movement_threshold, threshold_fast)

        label_counts = {}
        for lbl in track["labels"]:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        primary_label = max(label_counts, key=label_counts.get)

        avg_volume = sum(track["volumes"]) / len(track["volumes"]) if track["volumes"] else 0.0

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
            "avg_velocity": round(velocity_frame, 4),  # m/frame
            "avg_speed_m_per_s": round(float(speed_mps), 4) if speed_mps is not None else None,
            "direction": [round(d, 3) for d in direction],
            "movement_state": movement_state,
            "trajectory": positions,
            "timestamps": timestamps,
        }
        tracked_objects.append(track_summary)

        if movement_state != "stationary":
            movement_events.append(
                {
                    "track_id": track_id,
                    "label": primary_label,
                    "displacement": round(displacement, 2),
                    "speed_m_per_s": round(float(speed_mps), 3) if speed_mps is not None else None,
                    "direction": [round(d, 2) for d in direction],
                    "state": movement_state,
                }
            )

    print(f"         Found {len(movement_events)} significant movement events.")

    print("[STEP 4] Building spatial relationships...")

    def build_frame_edges(frame_objects, dist_thresh):
        edges = []
        for i in range(len(frame_objects)):
            for j in range(i + 1, len(frame_objects)):
                dist = calculate_distance(frame_objects[i]["centroid"], frame_objects[j]["centroid"])
                if dist < dist_thresh:
                    edges.append(
                        {
                            "source": frame_objects[i]["id"],
                            "target": frame_objects[j]["id"],
                            "distance": round(dist, 2),
                        }
                    )
        return edges

    first_frame_edges = build_frame_edges(all_frames[0], dist_threshold) if all_frames[0] else []
    last_frame_edges = build_frame_edges(all_frames[-1], dist_threshold) if all_frames[-1] else []

    approx_dt = None
    if frame_deltas_for_summary:
        approx_dt = float(sum(frame_deltas_for_summary) / len(frame_deltas_for_summary))

    temporal_graph = {
        "summary": {
            "total_frames": len(all_frames),
            "total_tracks": len(tracked_objects),
            "moving_objects": len(movement_events),
            "stationary_objects": len(tracked_objects) - len(movement_events),
            "approx_frame_interval_seconds": round(approx_dt, 3) if approx_dt is not None else None,
        },
        "tracked_objects": tracked_objects,
        "movement_events": movement_events,
        "spatial_relationships": {
            "first_frame": first_frame_edges,
            "last_frame": last_frame_edges,
        },
    }

    with open(output_graph_file, "w", encoding="utf-8") as f:
        json.dump(temporal_graph, f, indent=2)
    print(f"\n[SUCCESS] Temporal graph saved to: {output_graph_file}")

    print("[STEP 5] Generating temporal LLM prompt...")
    prompt = generate_temporal_prompt(temporal_graph, all_frames)
    with open(output_prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"[SUCCESS] Temporal prompt saved to: {output_prompt_file}")

    return output_prompt_file


def generate_temporal_prompt(temporal_graph, all_frames):
    """Generate a prompt for LLM temporal analysis."""
    summary = temporal_graph["summary"]
    tracked = temporal_graph["tracked_objects"]
    movements = temporal_graph["movement_events"]

    approx_dt = summary.get("approx_frame_interval_seconds")
    cadence_text = f"about {approx_dt} seconds between sampled frames" if approx_dt is not None else "unknown frame interval"

    prompt = f"""I have a TEMPORAL 3D scene graph from an underground longwall mine face.
The data spans {summary['total_frames']} consecutive point cloud frames with {cadence_text}.

Your task is to:
1. Identify mining equipment roles (Shearer, Hydraulic Chock, Conveyor, Human, etc.) from semantics + geometry + movement.
2. Analyze movement patterns to explain scene behavior over time.
3. Assess safety considering dynamic movements and proximity risks.

### Temporal Summary
- Total Frames Analyzed: {summary['total_frames']}
- Unique Objects Tracked: {summary['total_tracks']}
- Objects with Significant Movement: {summary['moving_objects']}
- Stationary Objects: {summary['stationary_objects']}

### Mining Context
1. Hydraulic Supports (Chocks): Large rectangular structures (~2m x 1.5m x 2m+), typically stationary in a row.
2. Shearer: Very large mobile machine (>8m3), expected to move along the coal face.
3. Segmentation classes provided: wall, equipment, human, conveyor, roof, other.
4. Safety rule: Humans or obstacles should not be in unsafe proximity to moving heavy equipment.

### Tracked Objects
"""

    for obj in tracked:
        prompt += f"\nTrack {obj['track_id']} ({obj['primary_label']}):\n"
        prompt += f"  - Avg volume: {obj['avg_volume']} m3\n"
        prompt += f"  - Frames: {obj['first_frame']} to {obj['last_frame']} ({obj['num_appearances']} appearances)\n"
        prompt += f"  - Movement state: {obj['movement_state']}\n"
        prompt += f"  - Displacement: {obj['total_displacement']} m\n"
        if obj.get("avg_speed_m_per_s") is not None:
            prompt += f"  - Avg speed: {obj['avg_speed_m_per_s']} m/s\n"
        prompt += f"  - Direction: {obj['direction']}\n"

    if movements:
        prompt += "\n### Significant Movement Events\n"
        for m in movements:
            speed_txt = f", speed {m['speed_m_per_s']} m/s" if m.get("speed_m_per_s") is not None else ""
            prompt += f"- Track {m['track_id']} ({m['label']}): moved {m['displacement']} m ({m['state']}{speed_txt})\n"

    prompt += """
### Output Requirements
Return ONLY valid JSON (no markdown, no extra text) with this structure:
{
  "object_identifications": [
    {
      "track_id": int,
      "primary_semantic_label": string,
      "identified_as": string,
      "confidence": "high|medium|low",
      "reasoning": string
    }
  ],
  "movement_analysis": string,
  "temporal_hazards": [
    {
      "type": string,
      "track_ids": [int],
      "severity": "low|medium|high",
      "reason": string
    }
  ],
  "temporal_safety_assessment": string
}
"""
    return prompt


if __name__ == "__main__":
    test_files = ["Results/01_test/frames/frame_00_objects.json"]
    build_temporal_graph(test_files, "Results/01_test")
