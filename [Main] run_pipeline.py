import os
import argparse
import sys

# Import functional modules
from func_segment_clustering import segment_geometry
from func_build_graph import build_scene_graph
from func_query_local_llm import query_local_llm
from func_visualize_scene_graph import visualize_graph

# ==========================================
#        CENTRALIZED CONFIGURATION
# ==========================================
class Config:
    # Input
    PCD_FILE = r"Datasets\pcd\all_downsampled_points.pcd"
    
    # Segmentation
    VOXEL_SIZE = 0.05
    RANSAC_DISTANCE = 0.1
    CLUSTER_EPS = 0.5
    CLUSTER_MIN_POINTS = 50
    
    # Graph
    GRAPH_DIST_THRESHOLD = 2.5
    
    # LLM
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def main():
    print(f"\n{'#'*60}")
    print(f"      3D SCENE GRAPH & LLM ANALYSIS PIPELINE")
    print(f"{'#'*60}\n")
    
    # 0. Setup Output Directory
    pcd_name = os.path.splitext(os.path.basename(Config.PCD_FILE))[0]
    results_dir = os.path.join("Results", pcd_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to: {results_dir}\n")
    
    # --- VISUALIZATION 1: RAW INPUT ---
    print("\n[VISUALIZATION] Showing Original Point Cloud...")
    import open3d as o3d
    try:
        raw_pcd = o3d.io.read_point_cloud(Config.PCD_FILE)
        o3d.visualization.draw_geometries([raw_pcd], window_name="1. Original Point Cloud (Raw) - Close to Continue")
    except Exception as e:
        print(f"[WARN] Could not visualize raw PCD: {e}")

    # 1. Geometric Segmentation
    print("\n>>> STARTING STAGE 1: GEOMETRIC SEGMENTATION")
    clustered_objects_file = segment_geometry(
        pcd_path=Config.PCD_FILE,
        output_dir=results_dir,
        voxel_size=Config.VOXEL_SIZE,
        ransac_dist=Config.RANSAC_DISTANCE,
        cluster_eps=Config.CLUSTER_EPS,
        min_points=Config.CLUSTER_MIN_POINTS,
        visualize=True # Show Step 2: Clusters
    )
    
    if not clustered_objects_file:
        print("[ABORT] Segmentation failed.")
        return

    # 2. Graph Construction
    print("\n>>> STARTING STAGE 2: GRAPH CONSTRUCTION")
    prompt_file = build_scene_graph(
        objects_file=clustered_objects_file,
        output_dir=results_dir,
        dist_threshold=Config.GRAPH_DIST_THRESHOLD
    )
    
    if not prompt_file:
        print("[ABORT] Graph construction failed.")
        return

    # 3. LLM Reasoning (Local DeepSeek)
    print("\n>>> STARTING STAGE 3: LLM REASONING")
    print("    (This may take a minute to load the model...)")
    llm_response_file = query_local_llm(
        prompt_file=prompt_file,
        output_dir=results_dir,
        model_name=Config.LLM_MODEL
    )
    
    if not llm_response_file:
        print("[ABORT] LLM inference failed.")
        return

    # 4. Visualization
    print("\n>>> STARTING STAGE 4: FINAL VISUALIZATION")
    print("[INFO] Showing Final Scene Graph + LLM Insights...")
    visualize_graph(
        pcd_path=Config.PCD_FILE,
        input_dir=results_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"      PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
