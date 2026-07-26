import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from segmentation.unet import UNet

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
            detections.append([int(x), int(y), int(x + w), int(y + h)])
            
    return detections

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize U-Net
    print("Initializing U-Net...")
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load('unet_carvana_scale0.5.pth'))
    model.eval()
    
    image_dir = 'images'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], 
                         key=lambda x: int(x.replace('image', '').replace('.png', '')))
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f"Starting segmentation on {len(image_files)} frames...")
    
    all_detections = {}

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        img_tensor = preprocess(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            # The model outputs 2 channels (background and foreground).
            # We take the argmax across the channel dimension to get the predicted class.
            mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
        detections = process_mask(mask)
        all_detections[img_name] = detections
        print(f"Segmented {img_name}: Found {len(detections)} cells.")
        
    # Store segmentation results
    with open("detections.json", "w") as f:
        json.dump(all_detections, f, indent=4)
        
    print("Segmentation complete. Bounding box detections saved to detections.json")

if __name__ == '__main__':
    main()
