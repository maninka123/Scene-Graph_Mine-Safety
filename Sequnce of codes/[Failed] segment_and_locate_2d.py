import os
# Fix for protobuf error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import json

# ==========================================
#              INPUT VARIABLES
# ==========================================
PCD_FILE_PATH = r"01\PC_20260122_153641662.pcd"
TEXT_QUERIES = [
    "shearer", "chock", "hydraulic support", 
    "floor", "rock", "machine", "equipment", "wall"
]
CONFIDENCE_THRESHOLD = 0.05
OUTPUT_JSON_FILE = "scene_objects.json"
OUTPUT_DEBUG_IMAGE = "projection_for_segmentation.png"
# ==========================================

def segment_and_extract_3d():
    print(f"\n{'='*40}")
    print(f"   Open-Vocabulary 3D Segmentation")
    print(f"{'='*40}\n")

    # 1. Load Data
    print(f"[STEP 1] Loading Point Cloud: {PCD_FILE_PATH}")
    pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 2. Project to 2D
    print(f"[STEP 2] Projecting {len(points):,} points to 2D image...")
    
    # Normalize for Front View (X-Z projection)
    x = points[:, 0]
    z = points[:, 2] 
    
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    
    width = 1000
    height = 1000
    
    # Map raw coords to image space
    u = ((x - x_min) / (x_max - x_min) * (width - 1)).astype(int)
    v = ((z - z_max) / (z_min - z_max) * (height - 1)).astype(int)
    
    img_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    pixel_to_indices = {}
    
    # Dilation settings
    dilation_size = 2 
    
    for i in range(len(points)):
        uc, vc = u[i], v[i]
        
        # Color mapping
        color_uint8 = (colors[i] * 255).astype(np.uint8)
        
        # Dilate for visibility (fill gaps)
        for du in range(-dilation_size, dilation_size+1):
            for dv in range(-dilation_size, dilation_size+1):
                ui, vi = uc + du, vc + dv
                if 0 <= ui < width and 0 <= vi < height:
                    img_buffer[vi, ui] = color_uint8
        
        # Store index map for back-projection (using exact center)
        key = (uc, vc)
        if key not in pixel_to_indices:
            pixel_to_indices[key] = []
        pixel_to_indices[key].append(i)

    image = Image.fromarray(img_buffer)
    image.save(OUTPUT_DEBUG_IMAGE)
    print(f"         Saved debug projection to {OUTPUT_DEBUG_IMAGE}")
    
    # Display the image for the user
    print(f"[DEBUG] Displaying 2D projection... (Close window to continue)")
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_buffer)
        plt.title("2D Input for Segmentation Model (X-Z Front View)")
        plt.axis('off')
        plt.show() # This blocks execution until closed
    except Exception as e:
        print(f"[WARNING] Could not display image: {e}")
    
    # 3. Model Inference
    print(f"[STEP 3] Running OWL-ViT Model...")
    print(f"         Queries: {TEXT_QUERIES}")
    
    try:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    inputs = processor(text=[TEXT_QUERIES], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        
    target_sizes = torch.Tensor([image.size[::-1]])
    # Try using the grounded detection method for newer transformers
    try:
        results = processor.post_process_grounded_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes, text_labels=[TEXT_QUERIES])[0]
    except AttributeError:
         # Fallback for older versions
        results = processor.post_process_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes)[0]
    
    # 4. Back-projection
    objects_3d = []
    print(f"[STEP 4] Back-projecting detections to 3D...")
    
    for i, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
        xmin, ymin, xmax, ymax = box.tolist()
        label_text = TEXT_QUERIES[label]
        
        # Find 3D points corresponding to pixels in the 2D box
        indices_in_box = []
        for ui in range(int(xmin), int(xmax)):
            for vi in range(int(ymin), int(ymax)):
                if (ui, vi) in pixel_to_indices:
                    indices_in_box.extend(pixel_to_indices[(ui, vi)])
        
        if not indices_in_box:
            continue
            
        # Compute 3D properties
        points_in_object = points[indices_in_box]
        centroid = np.mean(points_in_object, axis=0)
        
        obj_data = {
            "id": i,
            "label": label_text,
            "confidence": float(score),
            "centroid": centroid.tolist(),
            "point_count": len(indices_in_box)
        }
        objects_3d.append(obj_data)
        print(f"   > Found '{label_text}' (Conf: {score:.2f}) at {np.round(centroid, 2)}")

    # 5. Save Results
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(objects_3d, f, indent=2)
    print(f"\n[SUCCESS] Saved {len(objects_3d)} objects to {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    segment_and_extract_3d()

