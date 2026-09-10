
import os
import cv2
import re
import json
import numpy as np
import torch

from cellpose import models, io

# ---------------------------------------------------------------------------
# NOTE on weights: Cellpose ships its own pretrained checkpoints (not the
# same ecosystem as StarDist / cellseg_models_pytorch). The generalist
# checkpoint below ("cpsam", Cellpose-SAM, aka the default model in
# Cellpose 4+) works well across brightfield, phase-contrast, and
# fluorescence images without picking a tissue-specific checkpoint.
# ---------------------------------------------------------------------------
MODEL_TYPE = "cyto3"  # generalist model; good default for RBC / phase-contrast images

# Cellpose estimates cell diameter automatically if left as None. Set a
# fixed value (in pixels) if you know your average cell size and want to
# skip the (slower) auto-estimation step, or if auto-estimation is
# unreliable on your images.
DIAMETER = None


def process_masks(masks, area_threshold=20):
    detections = []
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    for label_id in unique_labels:
        ys, xs = np.where(masks == label_id)
        area = len(xs)
        if area <= area_threshold:
            continue

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        detections.append([x0, y0, x1, y1])

    return detections


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Cellpose model ({MODEL_TYPE}) on {device}...")

    model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type=MODEL_TYPE)

    image_dir = "CellsU"
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found.")
        return

    # image_files = sorted(
    #         [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)],
    valid_extensions = (".tif", ".tiff")
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)],
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", x)
        ],
    )

    print(f"Starting Cellpose segmentation on {len(image_files)} frames...")

    all_detections = {}

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)

            img = io.imread(img_path)  # returns HWC (RGB) or HW (grayscale)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

            masks, flows, styles = model.eval(
                img_gray,
                diameter=15,          
                channels=[0, 0],       
                flow_threshold=0.4,    
                cellprob_threshold=-1.0, 
                resample=False
            )

            # INDENTED INSIDE THE LOOP:
            detections = process_masks(masks)
            all_detections[img_name] = detections
            print(f"Segmented {img_name}: Found {len(detections)} cells.")

            if device == "cuda":
                torch.cuda.empty_cache()

    output_file = "detections.json"
    with open(output_file, "w") as f:
        json.dump(all_detections, f, indent=4)

    print(f"Segmentation complete. Bounding box detections saved to {output_file}")


if __name__ == "__main__":
    main()
    