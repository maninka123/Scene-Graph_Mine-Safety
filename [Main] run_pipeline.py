import os
from typing import Optional

from func_build_graph import build_scene_graph
from func_query_local_llm import query_local_llm
from func_segment_clustering import segment_geometry as segment_geometry_clustering
from func_segment_minkunet import segment_geometry_minkunet
from func_visualize_scene_graph import visualize_graph


class Config:
    # Input
    PCD_FILE = "Datasets/Data_all/PC_20260122_153641662.pcd"

    # Segmentation backend: "minkunet" or "clustering"
    SEGMENTATION_BACKEND = "minkunet"
    VISUALIZE_SEGMENTATION = True

    # MinkUNET segmentation
    MINKUNET_CONFIG = "MinkUNET/configs/inference.yaml"
    MINKUNET_CHECKPOINT = "MinkUNET/checkpoints/semantic_best.pt"
    MINKUNET_TEMPORAL_WINDOW = 1
    MINKUNET_CLASS_DBSCAN_EPS = None
    MINKUNET_CLASS_MIN_POINTS = None

    # Geometric clustering fallback
    VOXEL_SIZE = 0.05
    RANSAC_DISTANCE = 0.1
    CLUSTER_EPS = 0.5
    CLUSTER_MIN_POINTS = 50

    # Graph
    GRAPH_DIST_THRESHOLD = 2.5

    # LLM
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Visualization
    SHOW_RAW_INPUT = True


def run_segmentation(results_dir: str) -> Optional[str]:
    backend = str(Config.SEGMENTATION_BACKEND).strip().lower()
    if backend == "minkunet":
        print("\n>>> STARTING STAGE 1: SEMANTIC SEGMENTATION (MINKUNET)")
        return segment_geometry_minkunet(
            pcd_path=Config.PCD_FILE,
            output_dir=results_dir,
            config_path=Config.MINKUNET_CONFIG,
            checkpoint_path=Config.MINKUNET_CHECKPOINT,
            temporal_window=Config.MINKUNET_TEMPORAL_WINDOW,
            visualize=Config.VISUALIZE_SEGMENTATION,
            class_dbscan_eps=Config.MINKUNET_CLASS_DBSCAN_EPS,
            class_min_points=Config.MINKUNET_CLASS_MIN_POINTS,
        )

    if backend == "clustering":
        print("\n>>> STARTING STAGE 1: GEOMETRIC SEGMENTATION (CLUSTERING)")
        return segment_geometry_clustering(
            pcd_path=Config.PCD_FILE,
            output_dir=results_dir,
            voxel_size=Config.VOXEL_SIZE,
            ransac_dist=Config.RANSAC_DISTANCE,
            cluster_eps=Config.CLUSTER_EPS,
            min_points=Config.CLUSTER_MIN_POINTS,
            visualize=Config.VISUALIZE_SEGMENTATION,
        )

    print(f"[ABORT] Unsupported SEGMENTATION_BACKEND: {Config.SEGMENTATION_BACKEND}")
    return None


def main():
    print(f"\n{'#' * 60}")
    print("      3D SCENE GRAPH & LLM ANALYSIS PIPELINE")
    print(f"{'#' * 60}\n")

    pcd_name = os.path.splitext(os.path.basename(Config.PCD_FILE))[0]
    results_dir = os.path.join("Results", pcd_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Segmentation backend: {Config.SEGMENTATION_BACKEND}")
    print(f"[INFO] Results will be saved to: {results_dir}\n")

    if Config.SHOW_RAW_INPUT:
        print("\n[VISUALIZATION] Showing Original Point Cloud...")
        import open3d as o3d

        try:
            raw_pcd = o3d.io.read_point_cloud(Config.PCD_FILE)
            o3d.visualization.draw_geometries(
                [raw_pcd],
                window_name="1. Original Point Cloud (Raw) - Close to Continue",
            )
        except Exception as exc:
            print(f"[WARN] Could not visualize raw PCD: {exc}")

    segmented_objects_file = run_segmentation(results_dir=results_dir)
    if not segmented_objects_file:
        print("[ABORT] Segmentation failed.")
        return

    print("\n>>> STARTING STAGE 2: GRAPH CONSTRUCTION")
    prompt_file = build_scene_graph(
        objects_file=segmented_objects_file,
        output_dir=results_dir,
        dist_threshold=Config.GRAPH_DIST_THRESHOLD,
    )
    if not prompt_file:
        print("[ABORT] Graph construction failed.")
        return

    print("\n>>> STARTING STAGE 3: LLM REASONING")
    print("    (This may take a minute to load the model...)")
    llm_response_file = query_local_llm(
        prompt_file=prompt_file,
        output_dir=results_dir,
        model_name=Config.LLM_MODEL,
    )
    if not llm_response_file:
        print("[WARN] LLM inference did not return valid JSON. Continuing with graph-only visualization...")

    print("\n>>> STARTING STAGE 4: FINAL VISUALIZATION")
    print("[INFO] Showing Final Scene Graph + LLM Insights...")
    visualize_graph(
        pcd_path=Config.PCD_FILE,
        input_dir=results_dir,
    )

    print(f"\n{'#' * 60}")
    print("      PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
