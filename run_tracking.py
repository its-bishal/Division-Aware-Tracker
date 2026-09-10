import os
import cv2
import json
import numpy as np
from tracking.bipartite_matching import BiTracker

import re
import argparse
from utils.configs import TrackerHyperParams

def main():
    parser = argparse.ArgumentParser(description="Run cell tracking.")
    parser.add_argument('--scenario', type=str, default='model_b',
                        choices=['baseline', 'model_a', 'model_b', 'baseline_ukf', 'model_b_ukf'],
                        help=(
                            "Configuration scenario to run:\n"
                            "  baseline     - Standard Kalman filter only (no LDDMM)\n"
                            "  model_a      - Kalman + LDDMM (MSE loss)\n"
                            "  model_b      - Kalman + LDDMM (Sinkhorn OT loss)\n"
                            "  baseline_ukf - Unscented Kalman filter only (no LDDMM)\n"
                            "  model_b_ukf  - UKF + LDDMM (Sinkhorn OT loss)\n"
                        ))
    args = parser.parse_args()

    if not os.path.exists("detections.json"):
        print("Error: detections.json not found. Please run run_segmentation.py first.")
        return
        
    with open("detections.json", "r") as f:
        all_detections = json.load(f)

    print(f"Loading configuration for scenario: {args.scenario}")
    config = TrackerHyperParams(args.scenario)
    
    print("Initializing Bipartite Tracker...")
    tracker = BiTracker(config=config)
    
    image_dir = 'CellsU' #'images'
    # Ensure we iterate in the exact same numerical order
    # image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], 
    #                      key=lambda x: int(x.replace('image', '').replace('.png', '')))
    valid_extensions = (".tif", ".tiff")
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)],
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", x)
        ],
    )
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3)).tolist()

    out_dir = 'output_tracking'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting tracking across {len(image_files)} frames...")
    
    all_tracks_csv = []

    for frame_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        display_frame = frame.copy()
        
        detections = all_detections.get(img_name, [])
        detections_np = np.array(detections) if len(detections) > 0 else np.empty((0, 4))
        
        h, w = frame.shape[:2]
        masks = []
        for det in detections:
            mask = np.zeros((h, w), dtype=np.uint8)
            x_min, y_min, x_max, y_max = map(int, det[:4])
            mask[y_min:y_max, x_min:x_max] = 1
            masks.append(mask)
        masks_np = np.array(masks) if len(masks) > 0 else np.empty((0, h, w))
        
        tracker.update_track(frame_idx, [h, w], detections_np, masks_np)
        
        active_tracks = []
        for t in tracker.tracks.tracks:
            if t.missed_frame == 0 and len(t.nodes) > 0:
                bbox = t.nodes[-1].bbox
                active_tracks.append([bbox[0], bbox[1], bbox[2], bbox[3], t.track_id])
                
                # Save to CSV format: frame, id, x, y, w, h, 1, -1, -1, -1
                # (Assuming 1-indexed frames for standard MOT format)
                w_box = bbox[2] - bbox[0]
                h_box = bbox[3] - bbox[1]
                all_tracks_csv.append(f"{frame_idx + 1},{t.track_id},{bbox[0]},{bbox[1]},{w_box},{h_box},1,-1,-1,-1\n")
        
        for track in active_tracks:
            x_min, y_min, x_max, y_max, track_id = map(int, track)
            color = colors[track_id % len(colors)]
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(display_frame, f"ID: {track_id}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        print(f"Tracked {img_name}: {len(active_tracks)} active tracks")
        
        cv2.imwrite(os.path.join(out_dir, img_name), display_frame)

    with open("tracks.csv", "w") as f:
        f.writelines(all_tracks_csv)

    print(f"Tracking complete. Visualizations saved in {out_dir}/")
    print("Tracking coordinates saved to tracks.csv")


if __name__ == '__main__':
    main()

