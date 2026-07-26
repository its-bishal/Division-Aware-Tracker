import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from segmentation.unet import UNet
from tracking.bipartite_matching import BipartiteTracker

def process_mask(mask, threshold=0.5):
    """
    Process the U-Net mask to extract bounding boxes.
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    detections = []
    # start from 1 to ignore background label (0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Ignore very small detections
        if area > 20: 
            detections.append([x, y, x + w, y + h])
            
    return detections

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize U-Net
    print("Initializing U-Net...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    # If you have weights, load them like:
    # model.load_state_dict(torch.load('unet_weights.pth'))
    model.eval()
    
    # 2. Initialize Tracker
    tracker = BipartiteTracker(max_age=3, max_distance=50)
    
    image_dir = 'images'
    # Sort images numerically
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], 
                         key=lambda x: int(x.replace('image', '').replace('.png', '')))
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f"Starting tracking on {len(image_files)} frames...")
    
    # Define color for tracking boxes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3)).tolist()

    # Optional: Save output frames if needed (e.g. to a directory)
    out_dir = 'output_tracking'
    os.makedirs(out_dir, exist_ok=True)

    for frame_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        
        # Read frame
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Keep original frame for visualization
        display_frame = frame.copy()
        
        # Convert to tensor and add batch dimension
        img_tensor = preprocess(frame).unsqueeze(0).to(device)
        
        # Predict mask
        with torch.no_grad():
            output = model(img_tensor)
            mask = output.squeeze().cpu().numpy()
            
        # Process mask to bounding boxes
        detections = process_mask(mask)
        
        # Tracking Prediction Step
        tracker.predict()
        
        # Tracking Update Step
        tracker.update(detections)
        
        # Get active tracks for visualization
        active_tracks = tracker.get_active_tracks()
        
        # Draw bounding boxes
        for track in active_tracks:
            x_min, y_min, x_max, y_max, track_id = map(int, track)
            color = colors[track_id % len(colors)]
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(display_frame, f"ID: {track_id}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        print(f"Processed {img_name}: {len(active_tracks)} active tracks")
        
        # Save output
        cv2.imwrite(os.path.join(out_dir, img_name), display_frame)

    print(f"Tracking complete. Visualizations saved in {out_dir}/")

if __name__ == '__main__':
    main()
