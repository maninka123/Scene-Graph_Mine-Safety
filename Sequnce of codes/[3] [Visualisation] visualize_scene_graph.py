import open3d as o3d
import json
import numpy as np
import os

# ==========================================
#              INPUT VARIABLES
# ==========================================
PCD_FILE_PATH = r"01\PC_20260122_153641662.pcd"
GRAPH_FILE = "scene_graph.json"
LLM_RESPONSE_FILE = "mock_llm_response.json"
# ==========================================

def create_arrow(start_point, end_point, color=[1, 0.8, 0], radius=0.05):
    """
    Creates an arrow geometry from start_point to end_point.
    """
    start = np.array(start_point)
    end = np.array(end_point)
    vec = end - start
    length = np.linalg.norm(vec)
    
    if length < 0.001:
        return None

    # Create Arrow Geometry (Standard points up Z)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.0,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2,
        resolution=10,
        cylinder_split=1,
        cone_split=1
    )
    
    # Calculate Rotation to align Z-axis with 'vec'
    vec_norm = vec / length
    z_axis = np.array([0, 0, 1])
    
    # Rotation axis is cross product of Z and Target Vector
    rot_axis = np.cross(z_axis, vec_norm)
    rot_angle = np.arccos(np.dot(z_axis, vec_norm))
    
    # Handle parallel cases (no rotation needed or 180 flip)
    if np.linalg.norm(rot_axis) < 0.001:
        if np.dot(z_axis, vec_norm) < 0: # Pointing opposite
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

def visualize_graph():
    print(f"\n{'='*40}")
    print(f"   Scene Graph & LLM Insight Visualizer")
    print(f"{'='*40}\n")

    # 1. Load Point Cloud
    print(f"[STEP 1] Loading Point Cloud: {PCD_FILE_PATH}")
    pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    
    # 2. Load Graph
    print(f"[STEP 2] Loading Scene Graph: {GRAPH_FILE}")
    try:
        with open(GRAPH_FILE, "r") as f:
            graph = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {GRAPH_FILE}. Run [2] build_graph.py first.")
        return
        
    # 3. Load LLM Insights (Prefer Real, Fallback to Mock)
    REAL_RESPONSE = os.path.join("LLM", "llm_response_real.json")
    MOCK_RESPONSE = "mock_llm_response.json"
    
    if os.path.exists(REAL_RESPONSE):
        print(f"[STEP 3] Loading REAL LLM Insights: {REAL_RESPONSE}")
        target_file = REAL_RESPONSE
    else:
        print(f"[STEP 3] Loading MOCK LLM Insights: {MOCK_RESPONSE}")
        target_file = MOCK_RESPONSE
        
    try:
        with open(target_file, "r") as f:
            llm_data = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load LLM data: {e}")
        llm_data = {"insights": []}

    # Map LLM labels to IDs
    llm_labels = {item["object_id"]: item["generated_label"] for item in llm_data["insights"]}

    geometries = [pcd]
    
    # 4. Draw Nodes (Bounding Boxes)
    print(f"[STEP 4] Drawing Nodes (Objects)...")
    node_centers = {}
    
    for node in graph["nodes"]:
        center = np.array(node["properties"]["position"])
        dims = np.array(node["properties"]["dimensions"])
        node_id = node["id"]
        node_centers[node_id] = center
        
        # Create Bounding Box
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - dims/2, max_bound=center + dims/2)
        
        # Color Code:
        label = llm_labels.get(node_id, "Unknown")
        color = [0, 0, 1] # Blues/Purples for defaults
        
        if "Shearer" in label:
            color = [1, 0, 0] # Red
        elif "Chock" in label:
            color = [0, 1, 0] # Green
        elif "Unknown" in label:
             color = [0.5, 0.5, 0.5] # Grey
            
        bbox.color = color
        geometries.append(bbox)
        
        print(f"   > Node {node_id}: {label} (Color: {color})")

    # 5. Draw Edges (Arrows)
    print(f"[STEP 5] Drawing Edges (Relationships)...")
    arrow_count = 0
    
    for edge in graph["edges"]:
        source_id = edge["source"]
        target_id = edge["target"]
        
        if source_id in node_centers and target_id in node_centers:
            start = node_centers[source_id]
            end = node_centers[target_id]
            
            # Draw Arrow
            arrow = create_arrow(start, end, color=[1, 0.8, 0], radius=0.08)
            if arrow:
                geometries.append(arrow)
                arrow_count += 1
            
    print(f"   > Drew {arrow_count} connection arrows.")

    # 6. Visualize
    print(f"\n[INFO] Opening Visualization...")
    print(f"       RED BOX    = Shearer")
    print(f"       GREEN BOX  = Chocks")
    print(f"       YELLOW ARROW = Connection")
    
    print("\n--- LLM SAFETY REPORT ---")
    print(llm_data.get("safety_assessment", "No assessment."))
    print("-------------------------")
    
    o3d.visualization.draw_geometries(geometries, window_name="Scene Graph + LLM Insights")

if __name__ == "__main__":
    visualize_graph()
