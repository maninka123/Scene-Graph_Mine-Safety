import os

# Import temporal functional modules
from func_segment_clustering_temporal import segment_geometry_temporal
from func_build_temporal_graph import build_temporal_graph
from func_query_llm_temporal import query_llm_temporal
from func_visualize_temporal import visualize_temporal

# ==========================================
#        CENTRALIZED CONFIGURATION
# ==========================================
class TemporalConfig:
    # Dataset
    DATASET_FOLDER = r"Datasets\01"
    DATASET_NAME = "01"  # Used for output folder naming
    
    # Segmentation Parameters (same as single-frame pipeline)
    VOXEL_SIZE = 0.05
    RANSAC_DISTANCE = 0.1
    CLUSTER_EPS = 0.5
    CLUSTER_MIN_POINTS = 50
    
    # Temporal Tracking Parameters
    GRAPH_DIST_THRESHOLD = 2.5       # Max distance for "near" relationship
    MAX_TRACK_DISTANCE = 1.0         # Max centroid distance for object matching between frames
    MOVEMENT_THRESHOLD = 0.2         # Min displacement (m) to consider "moving"
    
    # LLM
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def main():
    print(f"\n{'#'*60}")
    print(f"      TEMPORAL 3D SCENE GRAPH PIPELINE")
    print(f"      Multi-Frame Movement Analysis")
    print(f"{'#'*60}\n")
    
    # 0. Setup Output Directory (named by dataset, not PCD file)
    results_dir = os.path.join("Results", TemporalConfig.DATASET_NAME)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Dataset: {TemporalConfig.DATASET_FOLDER}")
    print(f"[INFO] Results will be saved to: {results_dir}\n")
    
    # 1. Temporal Geometric Segmentation (Process All Frames)
    print("\n>>> STARTING STAGE 1: TEMPORAL SEGMENTATION")
    frame_files = segment_geometry_temporal(
        dataset_folder=TemporalConfig.DATASET_FOLDER,
        output_dir=results_dir,
        voxel_size=TemporalConfig.VOXEL_SIZE,
        ransac_dist=TemporalConfig.RANSAC_DISTANCE,
        cluster_eps=TemporalConfig.CLUSTER_EPS,
        min_points=TemporalConfig.CLUSTER_MIN_POINTS,
        visualize=False
    )
    
    if not frame_files:
        print("[ABORT] Temporal segmentation failed - no frames processed.")
        return
    
    print(f"[INFO] Successfully processed {len(frame_files)} frames.\n")

    # 2. Temporal Graph Construction (Object Tracking + Movement Analysis)
    print("\n>>> STARTING STAGE 2: TEMPORAL GRAPH CONSTRUCTION")
    prompt_file = build_temporal_graph(
        frame_files=frame_files,
        output_dir=results_dir,
        dist_threshold=TemporalConfig.GRAPH_DIST_THRESHOLD,
        max_track_distance=TemporalConfig.MAX_TRACK_DISTANCE,
        movement_threshold=TemporalConfig.MOVEMENT_THRESHOLD
    )
    
    if not prompt_file:
        print("[ABORT] Temporal graph construction failed.")
        return

    # 3. LLM Temporal Reasoning
    print("\n>>> STARTING STAGE 3: LLM TEMPORAL REASONING")
    print("    (This may take a minute to load the model...)")
    llm_response_file = query_llm_temporal(
        prompt_file=prompt_file,
        output_dir=results_dir,
        model_name=TemporalConfig.LLM_MODEL
    )
    
    if not llm_response_file:
        print("[WARN] LLM inference did not return valid JSON. Continuing with visualization...")

    # 4. Temporal Visualization
    print("\n>>> STARTING STAGE 4: TEMPORAL VISUALIZATION")
    print("[INFO] Showing Movement Trajectories + LLM Insights...")
    visualize_temporal(
        dataset_folder=TemporalConfig.DATASET_FOLDER,
        results_dir=results_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"      TEMPORAL PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#'*60}")
    print(f"\nResults saved in: {results_dir}")
    print(f"  - frames/          : Per-frame segmentation results")
    print(f"  - frame_index.json : Frame metadata")
    print(f"  - temporal_scene_graph.json : Object tracks & movements")
    print(f"  - LLM/             : Temporal LLM analysis")
    print(f"  - temporal_visualization.pcd : Visualized output")

if __name__ == "__main__":
    main()
