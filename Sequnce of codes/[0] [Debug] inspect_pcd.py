import open3d as o3d
import numpy as np
import os

# ==========================================
#              INPUT VARIABLES
# ==========================================
PCD_FILE_PATH = r"01\PC_20260122_153641662.pcd"
SHOW_VISUALIZATION = True  # Set to True to open the 3D viewer window
OUTPUT_INFO_FILE = "pcd_info.txt"
# ==========================================

def inspect_pcd():
    print(f"\n{'='*40}")
    print(f"   Point Cloud Inspection Tool")
    print(f"{'='*40}\n")

    # 1. Load File
    if not os.path.exists(PCD_FILE_PATH):
        print(f"[ERROR] File not found: {PCD_FILE_PATH}")
        return

    print(f"[INFO] Loading: {PCD_FILE_PATH}")
    try:
        pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load PCD: {e}")
        return

    # 2. Extract Metadata
    point_count = len(pcd.points)
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    center = pcd.get_center()

    # 3. Print Stats
    print(f"\n--- Statistics ---")
    print(f" > Point Count : {point_count:,}")
    print(f" > Has Colors  : {'Yes' if has_colors else 'No'}")
    print(f" > Has Normals : {'Yes' if has_normals else 'No'}")
    print(f" > Bounds      : \n\tMin: {min_bound}\n\tMax: {max_bound}")
    print(f" > Center      : {center}")

    if pcd.is_empty():
        print("\n[WARNING] Point cloud is empty!")

    # 4. Save to File
    with open(OUTPUT_INFO_FILE, "w") as f:
        f.write(f"File: {PCD_FILE_PATH}\n")
        f.write(f"Points: {point_count}\n")
        f.write(f"Bounds Min: {min_bound}\n")
        f.write(f"Bounds Max: {max_bound}\n")
    print(f"\n[INFO] Summary saved to: {OUTPUT_INFO_FILE}")

    # 5. Visualisation
    if SHOW_VISUALIZATION and not pcd.is_empty():
        print("\n[INFO] Opening 3D Visualization window...")
        print("       (Use Mouse to Rotate, Scroll to Zoom, 'Q' to Exit)")
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Inspection")
        print("[INFO] Visualization closed.")

if __name__ == "__main__":
    inspect_pcd()

