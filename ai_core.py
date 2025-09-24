import cv2
from ultralytics import YOLO

# This line loads the pre-trained AI model. 
# 'yolov8n.pt' is the nano version, which is the fastest.
model = YOLO('yolov8n.pt')
def process_video_file(input_path, output_path):
    """
    Processes a video file to detect objects and saves the output.
    """
    # --- 1. Open the video file ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # --- 2. Prepare the output video file ---
    # Get original video properties (width, height, frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to save the new video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # --- 3. Loop through every frame of the video ---
    while cap.isOpened():
        # Read one frame from the video
        ret, frame = cap.read()
        if not ret:
            # If no frame is returned, we've reached the end of the video
            break

        # --- 4. Run the AI on the frame ---
        # This is the magic line where the AI detection happens
        results = model(frame)

        # Use a built-in function to draw the detection boxes on the frame
        annotated_frame = results[0].plot()

        # Save the frame with the boxes to our new video file
        out.write(annotated_frame)

   # --- 5. Clean up ---
    # Release the video files to free up resources
    cap.release()
    out.release()
    print(f"Processing complete! Output video is saved at: {output_path}")

# This block should be OUTSIDE the function, with no indentation.
if __name__ == '__main__':
    process_video_file('test_video.mp4', 'output.mp4')