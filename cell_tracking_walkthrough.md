# Walkthrough: Cell Tracking using U-Net and Bipartite Matching

This guide outlines the step-by-step process for performing cell segmentation using U-Net and tracking using Bipartite Matching with the Kalman Filter in your `Cell-Tracker` project. We will use the dataset provided in the `images` folder (containing sequential frames `image1.png` to `image50.png`).

## Step 1: Data Preparation and Preprocessing
The images in the `images/` directory represent sequential frames of a time-lapse recording. 
1. **Load Images:** Read the images sequentially (e.g., using `cv2.imread` or `PIL.Image`).
2. **Preprocess:** Normalize the pixel values (e.g., scaling to [0, 1]), resize if necessary to match the input requirements of your U-Net model (e.g., $256 \times 256$ or $512 \times 512$), and convert them to PyTorch tensors.

## Step 2: Cell Segmentation using U-Net
The `segmentation/unet.py` file defines a foundational block `DoubleConv` for the U-Net architecture.
1. **Model Architecture Completion:** Ensure the full U-Net architecture is defined (using an encoder path with downsampling, a bottleneck, and a decoder path with upsampling and skip connections).
2. **Inference (Segmentation):** Pass each preprocessed frame through the trained U-Net model. The model will output a probability map (mask) indicating the presence of cells.
3. **Post-processing Masks:** 
   - Apply a threshold (e.g., $0.5$) to convert the probability map into a binary mask.
   - Run Connected Component Analysis (e.g., `cv2.connectedComponentsWithStats` or `skimage.measure.regionprops`) on the binary mask to separate individual cells.
   - Extract the **bounding boxes** (x_min, y_min, width, height) and **centroids** for each isolated cell component.

## Step 3: State Prediction with Kalman Filter
Your `tracking/kalman_filter.py` defines an 8-dimensional state space `[x, y, a, h, vx, vy, va, vh]`. The Kalman Filter helps in predicting where a cell will be in the *next* frame based on its current velocity.
1. **Initialization:** For the first frame, create a new track for each detected bounding box using the `initiate()` method.
2. **Prediction (Subsequent Frames):** Before analyzing the next frame, use the `predict()` method to estimate the new locations (bounding boxes) of all existing active tracks. This acts as our "Motion Gating" strategy mentioned in the `README.MD`.

## Step 4: Bipartite Matching (Hungarian Algorithm)
Now, we must associate the *predicted* tracks from the Kalman Filter with the *actual* detections (from U-Net) in the current frame.
1. **Cost Matrix Computation:** Construct a 2D matrix where rows represent existing tracks and columns represent current detections. The cost can be computed using a combination of metrics:
   - **Mahalanobis Distance:** Use `gating_distance()` from `kalman_filter.py` to get the spatial distance between the predicted bounding box and the detected bounding box.
   - **Intersection over Union (IoU):** Measure the overlap between the predicted bounding box and the new detection.
   - **Appearance/Shape Similarity:** If needed, compare the area or features of the cells.
2. **Solving the Assignment:** Use a bipartite matching algorithm (like the Hungarian algorithm via `scipy.optimize.linear_sum_assignment` in Python) on this cost matrix. This will globally optimize the assignments, ensuring one-to-one matching between predictions and detections with the minimum total cost.

## Step 5: State Update and Track Management
Once matches are found, we must update the Kalman Filter states and handle unassigned detections or lost tracks.
1. **Update Matches:** For every successful match, use the `update()` method in `kalman_filter.py` to correct the track's state using the actual detected bounding box.
2. **Unmatched Detections (New Cells):** If a new detection isn't matched to any track (e.g., a cell dividing or entering the frame), initialize a new Kalman Filter track for it.
3. **Unmatched Tracks (Lost Cells):** If a track isn't matched to any detection, mark it as "missed". If a track is missed for several consecutive frames, terminate it (the cell might have died, left the frame, or merged).
