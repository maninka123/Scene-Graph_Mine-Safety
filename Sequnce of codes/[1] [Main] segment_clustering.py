import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ==========================================
#              INPUT VARIABLES
# ==========================================
PCD_FILE_PATH = r"01\PC_20260122_153641662.pcd"
OUTPUT_JSON_FILE = "scene_objects_clustered.json"

# Algorithms Parameters
VOXEL_SIZE = 0.05           # Downsample size (meters)
RANSAC_DISTANCE = 0.1       # Distance threshold for floor detection (meters)
CLUSTER_EPS = 0.5           # Max distance between points in a cluster (meters)
CLUSTER_MIN_POINTS = 50     # Min points to form a cluster (after downsampling)
# ==========================================

def segment_geometry():
    print(f"\n{'='*40}")
    print(f"   Geometric Segmentation (Clustering)")
    print(f"{'='*40}\n")

    # 1. Load Data
    print(f"[STEP 1] Loading Point Cloud: {PCD_FILE_PATH}")
    if not os.path.exists(PCD_FILE_PATH):
        print(f"[ERROR] File not found.")
        return

    pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    print(f"         Loaded {len(pcd.points):,} points.")

    # 2. Downsample (for speed)
    print(f"[STEP 2] Downsampling (Voxel Grid: {VOXEL_SIZE}m)...")
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"         Reduced to {len(pcd_down.points):,} points.")

    # 3. Floor Removal (RANSAC)
    # We assume the largest plane is the floor/roof/wall and remove it to isolate objects
    print(f"[STEP 3] Removing dominant planes (Floor/Wall)...")
    
    objects_pcd = pcd_down
    floor_segments = []

    for i in range(2): 
        plane_model, inliers = objects_pcd.segment_plane(distance_threshold=RANSAC_DISTANCE,
                                                         ransac_n=3,
                                                         num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"         Plane {i+1}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        
        # Extract Inliers (Floor/Wall)
        inlier_cloud = objects_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0.4, 0.0]) # Orange
        floor_segments.append(inlier_cloud)

        # Extract Outliers (Objects)
        objects_pcd = objects_pcd.select_by_index(inliers, invert=True)
    
    print(f"         Remaining points (Objects): {len(objects_pcd.points):,}")

    # 4. Clustering (DBSCAN)
    print(f"[STEP 4] Clustering Objects (DBSCAN eps={CLUSTER_EPS}, min={CLUSTER_MIN_POINTS})...")
    labels = np.array(objects_pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=True))
    
    max_label = labels.max()
    print(f"         Found {max_label + 1} clusters.")

    # Color the clusters for visualization (excluding orange-ish colors conceptually, or just use tab20)
    # tab20 has orange at index 2/3, lets shift or just use jet
    colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Black for noise
    objects_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 5. Review & Visualization
    print(f"[STEP 5] Opening Visualization... (Close window to save data)")
    # Combine floor and objects for visualization
    vis_geometries = [objects_pcd] + floor_segments
    o3d.visualization.draw_geometries(vis_geometries, window_name="Clustered Objects + Floor (Orange)")

    # 6. Extract Object Data
    print(f"[STEP 6] Extracting object properties...")
    detected_objects = []
    
    for i in range(max_label + 1):
        # Extract indices for this cluster
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = objects_pcd.select_by_index(cluster_indices)
        
        # Bounding Box
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent() # Width, Height, Depth
        volume = extent[0] * extent[1] * extent[2]
        
        # Simple Logic to guess label (Can be improved with LLM)
        label = "unknown_object"
        if volume > 5.0:
            label = "large_machine_or_structure"
        elif extent[1] > 1.0: # Tall
            label = "potential_support"
            
        obj_data = {
            "id": int(i),
            "label": label,
            "confidence": 1.0, # Geometric certainty
            "centroid": center.tolist(),
            "dimensions": extent.tolist(),
            "point_count": len(cluster_indices)
        }
        detected_objects.append(obj_data)
        print(f"   > Cluster {i}: {label} at {np.round(center, 1)} (Vol: {volume:.1f}m3)")

    # 7. Save to JSON
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(detected_objects, f, indent=2)
    print(f"\n[SUCCESS] Saved {len(detected_objects)} clusters to {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    segment_geometry()
