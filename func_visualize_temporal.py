import open3d as o3d
import json
import numpy as np
import os
import matplotlib.pyplot as plt

def create_arrow(start_point, end_point, color=[1, 0.8, 0], radius=0.05):
    """Creates an arrow geometry from start_point to end_point."""
    start = np.array(start_point)
    end = np.array(end_point)
    vec = end - start
    length = np.linalg.norm(vec)
    
    if length < 0.001:
        return None

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.0,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2,
        resolution=10,
        cylinder_split=1,
        cone_split=1
    )
    
    vec_norm = vec / length
    z_axis = np.array([0, 0, 1])
    
    rot_axis = np.cross(z_axis, vec_norm)
    rot_angle = np.arccos(np.clip(np.dot(z_axis, vec_norm), -1.0, 1.0))
    
    if np.linalg.norm(rot_axis) < 0.001:
        if np.dot(z_axis, vec_norm) < 0:
            R = arrow.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
        else:
            R = np.eye(3)
    else:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R = arrow.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
    
    arrow.rotate(R, center=np.array([0, 0, 0]))
    arrow.translate(start)
    arrow.paint_uniform_color(color)
    return arrow

def create_trajectory_line(positions, color=[0, 1, 1]):
    """Create a line set showing object trajectory."""
    if len(positions) < 2:
        return None
    
    points = [np.array(p) for p in positions]
    lines = [[i, i+1] for i in range(len(points)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    
    return line_set

def visualize_temporal(dataset_folder, results_dir):
    """
    Visualizes temporal scene graph with movement trajectories.
    """
    print(f"\n{'='*50}")
    print(f"   [Function] Temporal Scene Graph Visualizer")
    print(f"{'='*50}\n")

    # Paths
    temporal_graph_file = os.path.join(results_dir, "temporal_scene_graph.json")
    llm_response_file = os.path.join(results_dir, "LLM", "llm_temporal_response.json")
    frame_index_file = os.path.join(results_dir, "frame_index.json")

    # 1. Load Temporal Graph
    print(f"[STEP 1] Loading Temporal Graph: {temporal_graph_file}")
    try:
        with open(temporal_graph_file, "r") as f:
            temporal_graph = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Temporal graph not found: {temporal_graph_file}")
        return

    tracked_objects = temporal_graph.get("tracked_objects", [])
    summary = temporal_graph.get("summary", {})
    
    print(f"         Loaded {len(tracked_objects)} tracked objects.")

    # 2. Load LLM Response
    print(f"[STEP 2] Loading LLM Temporal Insights: {llm_response_file}")
    llm_data = {}
    try:
        if os.path.exists(llm_response_file):
            with open(llm_response_file, "r") as f:
                llm_data = json.load(f)
        else:
            print(f"         [WARN] LLM response not found.")
    except Exception as e:
        print(f"         [WARN] Could not load LLM data: {e}")

    # Map LLM identifications to track IDs
    llm_identifications = {}
    for item in llm_data.get("object_identifications", []):
        track_id = item.get("track_id")
        if track_id is not None:
            llm_identifications[track_id] = item.get("identified_as", "Unknown")

    # 3. Load a reference point cloud (first frame with data)
    print(f"[STEP 3] Loading Reference Point Cloud...")
    reference_pcd = None
    
    try:
        with open(frame_index_file, "r") as f:
            frame_index = json.load(f)
        
        for frame in frame_index.get("frames", []):
            src_file = frame.get("source_file")
            if src_file and os.path.exists(src_file):
                reference_pcd = o3d.io.read_point_cloud(src_file)
                reference_pcd = reference_pcd.voxel_down_sample(voxel_size=0.1)
                print(f"         Loaded: {os.path.basename(src_file)}")
                break
    except Exception as e:
        print(f"         [WARN] Could not load reference PCD: {e}")

    geometries = []
    if reference_pcd:
        # Make background PCD semi-transparent gray
        reference_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(reference_pcd)

    # 4. Draw Trajectories and Movement Arrows
    print(f"[STEP 4] Drawing Object Trajectories...")
    
    # Color scheme based on movement state
    color_map = {
        "stationary": [0.3, 0.3, 0.8],   # Blue
        "moving_slow": [1.0, 0.8, 0.0],  # Yellow
        "moving_fast": [1.0, 0.0, 0.0]   # Red
    }
    
    # Color scheme based on LLM identification
    label_color_map = {
        "Shearer": [1.0, 0.0, 0.0],      # Red
        "Hydraulic Chock": [0.0, 0.8, 0.0],  # Green
        "Chock": [0.0, 0.8, 0.0],        # Green
        "Debris": [0.8, 0.5, 0.0],       # Orange
        "Unknown": [0.5, 0.5, 0.5]       # Gray
    }
    
    for obj in tracked_objects:
        track_id = obj["track_id"]
        positions = obj.get("trajectory", [])
        movement_state = obj.get("movement_state", "stationary")
        
        # Get color based on LLM identification or movement state
        llm_label = llm_identifications.get(track_id, "Unknown")
        if llm_label in label_color_map:
            color = label_color_map[llm_label]
        else:
            color = color_map.get(movement_state, [0.5, 0.5, 0.5])
        
        # Draw trajectory line
        if len(positions) >= 2:
            trajectory = create_trajectory_line(positions, color)
            if trajectory:
                geometries.append(trajectory)
        
        # Draw bounding box at final position
        if positions:
            final_pos = np.array(positions[-1])
            dims = np.array(obj.get("dimensions", [1, 1, 1])) if "dimensions" in obj else np.array([1, 1, 1])
            
            # Create a small sphere at each trajectory point
            for i, pos in enumerate(positions):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                sphere.translate(np.array(pos))
                sphere.paint_uniform_color(color)
                geometries.append(sphere)
        
        # Draw movement arrow for moving objects
        if movement_state != "stationary" and len(positions) >= 2:
            start = positions[0]
            end = positions[-1]
            arrow = create_arrow(start, end, color=[1, 0, 1], radius=0.1)  # Magenta arrows
            if arrow:
                geometries.append(arrow)
        
        print(f"   > Track {track_id}: {llm_label} ({movement_state})")

    # 5. Save visualization
    output_pcd = os.path.join(results_dir, "temporal_visualization.pcd")
    
    # Combine meshes for saving
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            combined_mesh += geom
    
    if len(combined_mesh.vertices) > 0:
        combined_pcd = combined_mesh.sample_points_uniformly(number_of_points=10000)
        if reference_pcd:
            combined_pcd = reference_pcd + combined_pcd
        o3d.io.write_point_cloud(output_pcd, combined_pcd)
        print(f"\n[SUCCESS] Saved visualization to: {output_pcd}")

    # 6. Print LLM Report & Append to File
    print(f"\n{'='*50}")
    print(f"      TEMPORAL ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    summary_lines = []
    summary_lines.append(f"\n{'='*50}")
    summary_lines.append(f"      VISUALIZATION SUMMARY")
    summary_lines.append(f"{'='*50}")
    summary_lines.append(f"Frames Analyzed: {summary.get('total_frames', 'N/A')}")
    summary_lines.append(f"Objects Tracked: {summary.get('total_tracks', 'N/A')}")
    summary_lines.append(f"Moving Objects: {summary.get('moving_objects', 'N/A')}")
    summary_lines.append(f"Stationary Objects: {summary.get('stationary_objects', 'N/A')}")
    
    print(f"Frames Analyzed: {summary.get('total_frames', 'N/A')}")
    print(f"Objects Tracked: {summary.get('total_tracks', 'N/A')}")
    print(f"Moving Objects: {summary.get('moving_objects', 'N/A')}")
    print(f"Stationary Objects: {summary.get('stationary_objects', 'N/A')}")
    
    if "movement_analysis" in llm_data:
        print(f"\n[MOVEMENT ANALYSIS]:")
        print(f"  {llm_data['movement_analysis']}")
    
    if "temporal_safety_assessment" in llm_data:
        print(f"\n[SAFETY ASSESSMENT]:")
        print(f"  {llm_data['temporal_safety_assessment']}")
    
    print(f"\n{'='*50}")
    print(f"       LEGEND")
    print(f"{'='*50}")
    print(f"  RED BOX/SPHERE    = Shearer (or fast-moving)")
    print(f"  GREEN BOX/SPHERE  = Hydraulic Chocks")
    print(f"  BLUE SPHERE       = Stationary objects")
    print(f"  YELLOW SPHERE     = Slow-moving objects")
    print(f"  MAGENTA ARROW     = Movement direction")
    print(f"  CYAN LINE         = Object trajectory")
    print(f"{'='*50}\n")
    
    summary_lines.append(f"\n{'='*50}")
    summary_lines.append(f"       LEGEND")
    summary_lines.append(f"{'='*50}")
    summary_lines.append(f"  RED BOX/SPHERE    = Shearer (or fast-moving)")
    summary_lines.append(f"  GREEN BOX/SPHERE  = Hydraulic Chocks")
    summary_lines.append(f"  BLUE SPHERE       = Stationary objects")
    summary_lines.append(f"  YELLOW SPHERE     = Slow-moving objects")
    summary_lines.append(f"  MAGENTA ARROW     = Movement direction")
    summary_lines.append(f"  CYAN LINE         = Object trajectory")
    summary_lines.append(f"{'='*50}\n")
    
    # Append to existing report file
    report_file = os.path.join(results_dir, "LLM", "llm_analysis_report.txt")
    try:
        with open(report_file, "a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"[SUCCESS] Appended visual summary to: {report_file}")
    except Exception as e:
        print(f"[WARN] Could not append to report file: {e}")

    # 7. Open visualization
    print(f"[INFO] Opening 3D Visualization...")
    o3d.visualization.draw_geometries(geometries, window_name="Temporal Scene Graph - Movement Analysis")


if __name__ == "__main__":
    visualize_temporal(r"Datasets\01", "Results/01")
