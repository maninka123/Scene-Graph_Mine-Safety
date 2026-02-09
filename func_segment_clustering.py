import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def segment_geometry(pcd_path, output_dir, voxel_size=0.05, ransac_dist=0.1, cluster_eps=0.5, min_points=50, visualize=False):
    """
    Performs geometric segmentation on a point cloud.
    Returns the path to the saved clustered objects JSON.
    """
    print(f"\n{'='*40}")
    print(f"   [Function] Geometric Segmentation")
    print(f"{'='*40}\n")

    # 1. Output Paths
    output_json = os.path.join(output_dir, "scene_objects_clustered.json")
    
    # 2. Load Data
    print(f"[STEP 1] Loading Point Cloud: {pcd_path}")
    if not os.path.exists(pcd_path):
        print(f"[ERROR] File not found.")
        return None

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"         Loaded {len(pcd.points):,} points.")

    # 3. Downsample
    print(f"[STEP 2] Downsampling (Voxel Grid: {voxel_size}m)...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"         Reduced to {len(pcd_down.points):,} points.")

    # 4. Floor Removal
    print(f"[STEP 3] Removing dominant planes (Floor/Wall)...")
    objects_pcd = pcd_down
    floor_segments = []

    for i in range(2): 
        plane_model, inliers = objects_pcd.segment_plane(distance_threshold=ransac_dist,
                                                         ransac_n=3,
                                                         num_iterations=1000)
        
        inlier_cloud = objects_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0.4, 0.0]) # Orange
        floor_segments.append(inlier_cloud)

        objects_pcd = objects_pcd.select_by_index(inliers, invert=True)
    
    print(f"         Remaining points (Objects): {len(objects_pcd.points):,}")

    # 5. Clustering
    print(f"[STEP 4] Clustering Objects (DBSCAN eps={cluster_eps}, min={min_points})...")
    labels = np.array(objects_pcd.cluster_dbscan(eps=cluster_eps, min_points=min_points, print_progress=True))
    
    max_label = labels.max()
    print(f"         Found {max_label + 1} clusters.")
    
    # Color clusters
    colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Black for noise
    objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    if visualize:
        print(f"[INFO] Showing Clustered Segments (Close window to continue)...")
        vis_geometries = [objects_pcd] + floor_segments
        o3d.visualization.draw_geometries(vis_geometries, window_name="Intermediate Result: Clusters + Floor")
    print(f"[STEP 6] Extracting object properties...")
    detected_objects = []
    
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = objects_pcd.select_by_index(cluster_indices)
        
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        volume = extent[0] * extent[1] * extent[2]
        
        label = "unknown_object"
        if volume > 5.0:
            label = "large_machine_or_structure"
        elif extent[1] > 1.0:
            label = "potential_support"
            
        obj_data = {
            "id": int(i),
            "label": label,
            "confidence": 1.0, 
            "centroid": center.tolist(),
            "dimensions": extent.tolist(),
            "point_count": len(cluster_indices)
        }
        detected_objects.append(obj_data)
        print(f"   > Cluster {i}: {label} (Vol: {volume:.1f}m3)")

    # 7. Save to JSON
    with open(output_json, "w") as f:
        json.dump(detected_objects, f, indent=2)
    print(f"\n[SUCCESS] Saved clusters to {output_json}")
    
    # 8. Save Colored PCD (For GitHub/Visualization)
    colored_pcd_path = os.path.join(output_dir, "colored_clusters.pcd")
    o3d.io.write_point_cloud(colored_pcd_path, objects_pcd)
    print(f"[SUCCESS] Saved colored cluster PCD to {colored_pcd_path}")
    
    return output_json

if __name__ == "__main__":
    segment_geometry(r"01\PC_20260122_153641662.pcd", ".")
