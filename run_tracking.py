import os
import cv2
import json
import numpy as np
from tracking.bipartite_matching import BiTracker

def main():
    # Load previously saved detections
    if not os.path.exists("detections.json"):
        print("Error: detections.json not found. Please run run_segmentation.py first.")
        return
        
    with open("detections.json", "r") as f:
        all_detections = json.load(f)

    # Initialize Tracker
    print("Initializing Bipartite Tracker...")
    tracker = BiTracker()
    
    image_dir = 'images'
    # Ensure we iterate in the exact same numerical order
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], 
                         key=lambda x: int(x.replace('image', '').replace('.png', '')))
    
    # Define colors for tracking boxes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3)).tolist()

    out_dir = 'output_tracking'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting tracking across {len(image_files)} frames...")

    for frame_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        display_frame = frame.copy()
        
        # 1. Fetch the pre-computed detections for this frame
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
        
        # 2. Tracking Update Step
        tracker.update_track(frame_idx, [h, w], detections_np, masks_np)
        
        # 3. Get active tracks for visualization
        active_tracks = []
        for t in tracker.tracks.tracks:
            if t.missed_frame == 0 and len(t.nodes) > 0:
                bbox = t.nodes[-1].bbox
                active_tracks.append([bbox[0], bbox[1], bbox[2], bbox[3], t.track_id])
        
        # Draw bounding boxes
        for track in active_tracks:
            x_min, y_min, x_max, y_max, track_id = map(int, track)
            color = colors[track_id % len(colors)]
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(display_frame, f"ID: {track_id}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        print(f"Tracked {img_name}: {len(active_tracks)} active tracks")
        
        cv2.imwrite(os.path.join(out_dir, img_name), display_frame)

    print(f"Tracking complete. Visualizations saved in {out_dir}/")

if __name__ == '__main__':
    main()
