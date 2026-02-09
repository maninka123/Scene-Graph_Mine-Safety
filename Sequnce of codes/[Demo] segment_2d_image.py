import os
# Fix for protobuf error if apparent
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
IMAGE_PATH = "projection_view.png" # Using the image we just generated
TEXT_QUERIES = ["shearer", "chock", "hydraulic support", "floor", "rock"]
CONFIDENCE_THRESHOLD = 0.1

def detect_objects():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found. Run project_to_image.py first.")
        return

    print("Loading OWL-ViT model (this may take a moment to download weights)...")
    try:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Processing image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Prepare inputs
    inputs = processor(text=TEXT_QUERIES, images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes)[0]

    # Visualization
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    print(f"\nDetections (Threshold > {CONFIDENCE_THRESHOLD}):")
    found_any = False

    for i, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
        box = [round(i, 2) for i in box.tolist()]
        label_text = TEXT_QUERIES[label]
        
        print(f"  - Detected '{label_text}' with confidence {round(score.item(), 3)} at {box}")
        
        # Draw box
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0, f"{label_text}: {score:.2f}", color='white', fontsize=8, backgroundcolor='red')
        found_any = True

    if not found_any:
        print("  No objects detected. Try lowering threshold or changing queries.")

    plt.axis('off')
    output_file = "segmentation_result.png"
    plt.savefig(output_file)
    print(f"\nSaved annotated image to {output_file}")

if __name__ == "__main__":
    detect_objects()
