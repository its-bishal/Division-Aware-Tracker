import os
import cv2
import json
import re
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
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 20: 
            detections.append([int(x), int(y), int(x + w), int(y + h)])
            
    return detections


def sort_key(filename):
    """Natural sorting key: splits string into text and integer chunks."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]


def load_tiff_image(img_path):
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if frame is None:
        return None

    converters = {
        2: lambda img: cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
        3: lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        4: lambda img: cv2.cvtColor(img, cv2.COLOR_BGRA2RGB),
    }

    frame = converters.get(
        len(frame.shape),
        lambda img: img
    )(frame)

    normalization = {
        np.uint8: 1 / 255.0,
        np.uint16: 1 / 65535.0,
    }

    scale = normalization.get(frame.dtype.type, 1.0)

    return (frame * scale).astype(np.float32)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Initializing U-Net...")
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load('unet_carvana_scale0.5.pth'))
    model.eval()
    
    image_dir = 'images' #'images'
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], 
                         key=lambda x: int(x.replace('image', '').replace('.png', '')))
    
    # valid_exts = ('.tiff', '.tif', '.TIFF', '.TIF')
    # image_files = sorted(
    #     [f for f in os.listdir(image_dir) if f.endswith(valid_exts)],
    #     key=sort_key
    # )
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f"Starting segmentation on {len(image_files)} frames...")
    
    all_detections = {}

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        # frame = load_tiff_image(img_path)
        if frame is None:
            continue
            
        img_tensor = preprocess(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            mask = output.argmax(dim=1).squeeze().cpu().numpy()
            
        detections = process_mask(mask)
        all_detections[img_name] = detections
        print(f"Segmented {img_name}: Found {len(detections)} cells.")
        
    with open("detections.json", "w") as f:
        json.dump(all_detections, f, indent=4)
        
    print("Segmentation complete. Bounding box detections saved to detections.json")

if __name__ == '__main__':
    main()

