import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import re

def segment_geometry_temporal(dataset_folder, output_dir, voxel_size=0.05, ransac_dist=0.1, 
                               cluster_eps=0.5, min_points=50, visualize=False):
    """
    Processes multiple PCD frames from a dataset folder.
    Returns list of paths to per-frame clustered object JSON files.
    """
    print(f"\n{'='*50}")
    print(f"   [Function] Temporal Geometric Segmentation")
    print(f"{'='*50}\n")

    # 1. Create Output Directory
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 2. Find all PCD files and sort by timestamp
    print(f"[STEP 1] Scanning Dataset Folder: {dataset_folder}")
    pcd_pattern = os.path.join(dataset_folder, "PC_*.pcd")
    pcd_files = sorted(glob.glob(pcd_pattern))
    
    if not pcd_files:
        print(f"[ERROR] No PCD files found matching pattern: {pcd_pattern}")
        return None
    
    print(f"         Found {len(pcd_files)} PCD frames.")
    
    # 3. Extract timestamps from filenames
    def extract_timestamp(filepath):
        """Extract timestamp from filename like PC_20260122_153641662.pcd"""
        filename = os.path.basename(filepath)
        match = re.search(r'PC_(\d{8}_\d+)\.pcd', filename)
        if match:
            return match.group(1)
        return filename
    
    # 4. Process each frame
    all_frame_results = []
    frame_metadata = []
    
    for frame_idx, pcd_path in enumerate(pcd_files):
        timestamp = extract_timestamp(pcd_path)
        print(f"\n[FRAME {frame_idx:02d}] Processing: {os.path.basename(pcd_path)}")
        
        # Output paths for this frame
        frame_json = os.path.join(frames_dir, f"frame_{frame_idx:02d}_objects.json")
        frame_pcd = os.path.join(frames_dir, f"frame_{frame_idx:02d}_clusters.pcd")
        
        # Load point cloud
        if not os.path.exists(pcd_path):
            print(f"         [SKIP] File not found.")
            continue
            
        pcd = o3d.io.read_point_cloud(pcd_path)
        original_points = len(pcd.points)
        print(f"         Loaded {original_points:,} points.")
        
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"         Downsampled to {len(pcd_down.points):,} points.")
        
        # Floor/Wall Removal (RANSAC)
        objects_pcd = pcd_down
        for i in range(2):
            if len(objects_pcd.points) < 100:
                break
            plane_model, inliers = objects_pcd.segment_plane(
                distance_threshold=ransac_dist,
                ransac_n=3,
                num_iterations=1000
            )
            objects_pcd = objects_pcd.select_by_index(inliers, invert=True)
        
        print(f"         After plane removal: {len(objects_pcd.points):,} points.")
        
        # Clustering
        if len(objects_pcd.points) < min_points:
            print(f"         [SKIP] Not enough points for clustering.")
            continue
            
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=cluster_eps, 
            min_points=min_points, 
            print_progress=False
        ))
        
        max_label = labels.max()
        num_clusters = max_label + 1 if max_label >= 0 else 0
        print(f"         Found {num_clusters} clusters.")
        
        if num_clusters == 0:
            continue
        
        # Color clusters for visualization
        colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        # Extract object properties
        detected_objects = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = objects_pcd.select_by_index(cluster_indices)
            
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            volume = extent[0] * extent[1] * extent[2]
            
            # Basic labeling
            label = "unknown_object"
            if volume > 5.0:
                label = "large_machine_or_structure"
            elif extent[1] > 1.0:
                label = "potential_support"
            
            obj_data = {
                "id": int(i),
                "frame_id": frame_idx,
                "timestamp": timestamp,
                "label": label,
                "centroid": center.tolist(),
                "dimensions": extent.tolist(),
                "volume": volume,
                "point_count": len(cluster_indices)
            }
            detected_objects.append(obj_data)
        
        # Save frame results
        with open(frame_json, "w") as f:
            json.dump(detected_objects, f, indent=2)
        
        o3d.io.write_point_cloud(frame_pcd, objects_pcd)
        
        all_frame_results.append(frame_json)
        frame_metadata.append({
            "frame_id": frame_idx,
            "timestamp": timestamp,
            "source_file": pcd_path,
            "num_objects": len(detected_objects),
            "objects_file": frame_json
        })
        
        print(f"         Saved: {os.path.basename(frame_json)}")
    
    # 5. Save frame index
    index_file = os.path.join(output_dir, "frame_index.json")
    with open(index_file, "w") as f:
        json.dump({
            "total_frames": len(frame_metadata),
            "frames": frame_metadata
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"[SUCCESS] Processed {len(all_frame_results)} frames.")
    print(f"          Frame index saved to: {index_file}")
    print(f"{'='*50}\n")
    
    return all_frame_results


if __name__ == "__main__":
    segment_geometry_temporal(r"Datasets\01", "Results/01_test")
