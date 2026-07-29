
import os
import cv2
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
#
# If you want the older style-specific models instead, swap MODEL_TYPE for
# one of: "cyto3", "cyto2", "cyto", "nuclei". All are downloaded
# automatically on first use and cached in $HOME/.cellpose/models/.
# ---------------------------------------------------------------------------
MODEL_TYPE = "cpsam"  # generalist model; good default for RBC / phase-contrast images

# Cellpose estimates cell diameter automatically if left as None. Set a
# fixed value (in pixels) if you know your average cell size and want to
# skip the (slower) auto-estimation step, or if auto-estimation is
# unreliable on your images.
DIAMETER = None


def process_masks(masks, area_threshold=20):
    """
    Extract bounding boxes from a Cellpose instance label mask.
    `masks` is a 2D int array where each cell has a distinct integer label
    and background is 0 (this is Cellpose's native output format, so no
    connected-components step is needed the way it was for the StarDist
    polygon-to-raster conversion).
    """
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

    image_dir = "images"
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found.")
        return

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
        key=lambda x: int(x.replace("image", "").replace(".png", "")),
    )

    print(f"Starting Cellpose segmentation on {len(image_files)} frames...")

    all_detections = {}

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)

        img = io.imread(img_path)  # returns HWC (RGB) or HW (grayscale)

        # Cellpose does its own internal normalization (percentile-based,
        # similar in spirit to what the StarDist script did manually), so
        # no separate percentile_normalize() step is needed here.
        #
        # channels=[0, 0] tells Cellpose to treat the image as single-channel
        # grayscale (no separate nuclear channel) -- appropriate for
        # brightfield / phase-contrast RBC images. If you have a real
        # two-channel image (e.g. cytoplasm + nucleus stain), set this to
        # [1, 2] or similar per the Cellpose docs.
        masks, flows, styles = model.eval(
            img,
            diameter=DIAMETER,
            channels=[0, 0],
        )

        detections = process_masks(masks)

        all_detections[img_name] = detections
        print(f"Segmented {img_name}: Found {len(detections)} cells.")

    output_file = "detections.json"
    with open(output_file, "w") as f:
        json.dump(all_detections, f, indent=4)

    print(f"Segmentation complete. Bounding box detections saved to {output_file}")


if __name__ == "__main__":
    main()
    