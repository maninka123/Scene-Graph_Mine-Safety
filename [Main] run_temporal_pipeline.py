import os
from typing import Optional

from func_build_temporal_graph import build_temporal_graph
from func_query_llm_temporal import query_llm_temporal
from func_segment_clustering_temporal import segment_geometry_temporal as segment_geometry_temporal_clustering
from func_segment_minkunet_temporal import segment_geometry_temporal_minkunet
from func_visualize_temporal import visualize_temporal


class TemporalConfig:
    # Dataset
    DATASET_FOLDER = "Datasets/01"
    DATASET_NAME = "01"

    # Segmentation backend: "minkunet" or "clustering"
    SEGMENTATION_BACKEND = "minkunet"
    VISUALIZE_SEGMENTATION = False

    # MinkUNET temporal segmentation
    MINKUNET_CONFIG = "MinkUNET/configs/inference.yaml"
    MINKUNET_CHECKPOINT = "MinkUNET/checkpoints/semantic_best.pt"
    MINKUNET_TEMPORAL_WINDOW = 5
    MINKUNET_CLASS_DBSCAN_EPS = None
    MINKUNET_CLASS_MIN_POINTS = None

    # Geometric clustering fallback
    VOXEL_SIZE = 0.05
    RANSAC_DISTANCE = 0.1
    CLUSTER_EPS = 0.5
    CLUSTER_MIN_POINTS = 50

    # Temporal graph / tracking
    GRAPH_DIST_THRESHOLD = 2.5
    MAX_TRACK_DISTANCE = 1.0
    MOVEMENT_THRESHOLD = 0.2

    # LLM
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def run_temporal_segmentation(results_dir: str) -> Optional[list]:
    backend = str(TemporalConfig.SEGMENTATION_BACKEND).strip().lower()
    if backend == "minkunet":
        print("\n>>> STARTING STAGE 1: TEMPORAL SEMANTIC SEGMENTATION (MINKUNET)")
        return segment_geometry_temporal_minkunet(
            dataset_folder=TemporalConfig.DATASET_FOLDER,
            output_dir=results_dir,
            config_path=TemporalConfig.MINKUNET_CONFIG,
            checkpoint_path=TemporalConfig.MINKUNET_CHECKPOINT,
            temporal_window=TemporalConfig.MINKUNET_TEMPORAL_WINDOW,
            visualize=TemporalConfig.VISUALIZE_SEGMENTATION,
            class_dbscan_eps=TemporalConfig.MINKUNET_CLASS_DBSCAN_EPS,
            class_min_points=TemporalConfig.MINKUNET_CLASS_MIN_POINTS,
        )

    if backend == "clustering":
        print("\n>>> STARTING STAGE 1: TEMPORAL GEOMETRIC SEGMENTATION (CLUSTERING)")
        return segment_geometry_temporal_clustering(
            dataset_folder=TemporalConfig.DATASET_FOLDER,
            output_dir=results_dir,
            voxel_size=TemporalConfig.VOXEL_SIZE,
            ransac_dist=TemporalConfig.RANSAC_DISTANCE,
            cluster_eps=TemporalConfig.CLUSTER_EPS,
            min_points=TemporalConfig.CLUSTER_MIN_POINTS,
            visualize=TemporalConfig.VISUALIZE_SEGMENTATION,
        )

    print(f"[ABORT] Unsupported SEGMENTATION_BACKEND: {TemporalConfig.SEGMENTATION_BACKEND}")
    return None


def main():
    print(f"\n{'#' * 60}")
    print("      TEMPORAL 3D SCENE GRAPH PIPELINE")
    print("      Multi-Frame Movement Analysis")
    print(f"{'#' * 60}\n")

    results_dir = os.path.join("Results", "Temporal_Sequence", TemporalConfig.DATASET_NAME)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Dataset: {TemporalConfig.DATASET_FOLDER}")
    print(f"[INFO] Segmentation backend: {TemporalConfig.SEGMENTATION_BACKEND}")
    print(f"[INFO] Results will be saved to: {results_dir}\n")

    frame_files = run_temporal_segmentation(results_dir=results_dir)
    if not frame_files:
        print("[ABORT] Temporal segmentation failed - no frames processed.")
        return
    print(f"[INFO] Successfully processed {len(frame_files)} frames.\n")

    print("\n>>> STARTING STAGE 2: TEMPORAL GRAPH CONSTRUCTION")
    prompt_file = build_temporal_graph(
        frame_files=frame_files,
        output_dir=results_dir,
        dist_threshold=TemporalConfig.GRAPH_DIST_THRESHOLD,
        max_track_distance=TemporalConfig.MAX_TRACK_DISTANCE,
        movement_threshold=TemporalConfig.MOVEMENT_THRESHOLD,
    )
    if not prompt_file:
        print("[ABORT] Temporal graph construction failed.")
        return

    print("\n>>> STARTING STAGE 3: LLM TEMPORAL REASONING")
    print("    (This may take a minute to load the model...)")
    llm_response_file = query_llm_temporal(
        prompt_file=prompt_file,
        output_dir=results_dir,
        model_name=TemporalConfig.LLM_MODEL,
    )
    if not llm_response_file:
        print("[WARN] LLM inference did not return valid JSON. Continuing with visualization...")

    print("\n>>> STARTING STAGE 4: TEMPORAL VISUALIZATION")
    print("[INFO] Showing Movement Trajectories + LLM Insights...")
    visualize_temporal(
        dataset_folder=TemporalConfig.DATASET_FOLDER,
        results_dir=results_dir,
    )

    print(f"\n{'#' * 60}")
    print("      TEMPORAL PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#' * 60}")
    print(f"\nResults saved in: {results_dir}")
    print("  - frames/                    : Per-frame segmentation results")
    print("  - frame_index.json           : Frame metadata")
    print("  - temporal_scene_graph.json  : Object tracks & movements")
    print("  - LLM/                       : Temporal LLM analysis")
    print("  - temporal_visualization.pcd : Visualized output")


if __name__ == "__main__":
    main()
