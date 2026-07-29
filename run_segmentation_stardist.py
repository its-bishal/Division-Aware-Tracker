import os
import cv2
import json
import numpy as np
import torch

from albumentations import Resize, Compose
from cellseg_models_pytorch.models.stardist import StarDist
from cellseg_models_pytorch.utils import FileHandler

# ---------------------------------------------------------------------------
# NOTE on weights: cellseg_models_pytorch (csmp) is a *different* library
# from the original TensorFlow `stardist` package. There is no
# '2D_versatile_fluo' model here -- pretrained csmp StarDist checkpoints are
# hosted at https://huggingface.co/csmp-hub (e.g. "hgsc_v1_efficientnet_b5").
# Pick the checkpoint whose training domain best matches your images, or
# swap in a path to your own fine-tuned .ckpt/.pth if you have one.
# ---------------------------------------------------------------------------
WEIGHTS = "hgsc_v1_efficientnet_b5"

# csmp models require the spatial dims to be a multiple of 32.
RESIZE_TO = 1024


def percentile_normalize(img, pmin=1, pmax=99.8, eps=1e-8):
    """
    Pure-numpy re-implementation of csbdeep.utils.normalize's percentile
    normalization, so we don't need to pull in TensorFlow/csbdeep just for
    this one utility function.
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)
    out = (img - lo) / (hi - lo + eps)
    return np.clip(out, 0, 1).astype(np.float32)


def process_labels(labels, area_threshold=20):
    """
    Extract bounding boxes from a StarDist instance label mask.
    """
    detections = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for label_id in unique_labels:
        binary_mask = (labels == label_id).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        if num_labels > 1:
            x, y, w, h, area = stats[1]
            if area > area_threshold:
                detections.append([int(x), int(y), int(x + w), int(y + h)])

    return detections


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing StarDist model on {device}...")

    model = StarDist.from_pretrained(weights=WEIGHTS)
    model.set_inference_mode()
    
    try:
        model.to(device)
    except AttributeError:
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(device)
        elif hasattr(model, 'net') and hasattr(model.net, 'to'):
            model.net.to(device)
        else:
            print(f"Warning: Could not explicitly move model to {device}. It may run on CPU or handle it internally.")

    transform = Compose([Resize(RESIZE_TO, RESIZE_TO)])

    image_dir = "images"
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} not found.")
        return

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
        key=lambda x: int(x.replace("image", "").replace(".png", "")),
    )

    print(f"Starting StarDist segmentation on {len(image_files)} frames...")

    all_detections = {}

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)

        img_rgb = FileHandler.read_img(img_path)  # returns RGB, HWC, uint8
        orig_h, orig_w = img_rgb.shape[:2]

        img_norm = percentile_normalize(img_rgb, 1, 99.8)
        img_resized = transform(image=img_norm)["image"]

        with torch.no_grad():
            prob = model.predict(img_resized)
            out = model.post_process(prob)

        # out["nuc"] is a list (one entry per image in the batch) of
        # (instance_map, type_map) tuples.
        labels, _ = out["nuc"][0]

        # Scale bboxes back to the original image size, since we resized
        # to RESIZE_TO x RESIZE_TO for inference.
        scale_x = orig_w / RESIZE_TO
        scale_y = orig_h / RESIZE_TO

        detections = []
        for x0, y0, x1, y1 in process_labels(labels):
            detections.append([
                int(x0 * scale_x), int(y0 * scale_y),
                int(x1 * scale_x), int(y1 * scale_y),
            ])

        all_detections[img_name] = detections
        print(f"Segmented {img_name}: Found {len(detections)} cells.")

    output_file = "detections.json"
    with open(output_file, "w") as f:
        json.dump(all_detections, f, indent=4)

    print(f"Segmentation complete. Bounding box detections saved to {output_file}")


if __name__ == "__main__":
    main()