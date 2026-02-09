import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
#              INPUT VARIABLES
# ==========================================
PCD_FILE_PATH = r"01\PC_20260122_153641662.pcd"
OUTPUT_IMAGE_PATH = "projection_view.png"
DPI = 150
# ==========================================

def project_and_save():
    print(f"\n{'='*40}")
    print(f"      Point Cloud 2D Projector")
    print(f"{'='*40}\n")

    print(f"[INFO] Loading: {PCD_FILE_PATH}")
    if not os.path.exists(PCD_FILE_PATH):
        print(f"[ERROR] File not found.")
        return

    pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print(f"[INFO] Projecting {len(points):,} points to 2D...")

    # Create a figure with 3 subplots (Top, Front, Side)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Top View (X-Y)
    axs[0].scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.5)
    axs[0].set_title("Top View (X-Y)")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis('equal')
    axs[0].grid(True, linestyle='--', alpha=0.3)

    # 2. Front View (X-Z)
    axs[1].scatter(points[:, 0], points[:, 2], c=colors, s=1, alpha=0.5)
    axs[1].set_title("Front View (X-Z)")
    axs[1].set_xlabel("X (m)")
    axs[1].set_ylabel("Z (m)")
    axs[1].axis('equal')
    axs[1].grid(True, linestyle='--', alpha=0.3)

    # 3. Side View (Y-Z)
    axs[2].scatter(points[:, 1], points[:, 2], c=colors, s=1, alpha=0.5)
    axs[2].set_title("Side View (Y-Z)")
    axs[2].set_xlabel("Y (m)")
    axs[2].set_ylabel("Z (m)")
    axs[2].axis('equal')
    axs[2].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=DPI)
    print(f"[SUCCESS] Saved projections to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    project_and_save()

