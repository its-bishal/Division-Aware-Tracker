import cv2
import os

def create_video(image_folder, output_video, fps=5):
    # Get all images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    
    if not images:
        print(f"No images found in {image_folder}")
        return

    # Sort images by the numerical value in their filename to ensure correct order
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Read the first frame to get the height and width
    first_frame_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    # 'mp4v' is a common codec for generating .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Creating video '{output_video}' from {len(images)} frames...")
    
    # Iterate through all images and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer to finalize the file
    video.release()
    print(f"Video successfully saved to {output_video}")

if __name__ == "__main__":
    input_folder = 'output_tracking'
    output_file = 'tracking_result.mp4'
    
    create_video(input_folder, output_file, fps=5)
