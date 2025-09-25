import cv2
from ultralytics import YOLO

# This line loads the pre-trained AI model.
# 'yolov8n.pt' is the nano version, which is the fastest.
model = YOLO('yolov8n.pt')

# --- This function remains the same from Day 1 ---
def process_video_file(input_path, output_path):
    """
    Processes a video file to detect objects and saves the output.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a VideoWriter object to save the new video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    alerts = []
    frame_count = 0

    # Loop through every frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run the AI on the frame
        results = model(frame)
        
        # Use a built-in function to draw the detection boxes on the frame
        annotated_frame = results[0].plot()
        
        # Check for specific objects (e.g., 'person') and create alerts
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls)] == 'person':
                    timestamp = frame_count / fps
                    # We create a simple list of strings for the alerts
                    alerts.append(f"Person detected at {timestamp:.2f} seconds")

        # Save the frame with the boxes to our new video file
        out.write(annotated_frame)
        frame_count += 1

    # Release the video files to free up resources
    cap.release()
    out.release()
    print(f"Processing complete! Output video is saved at: {output_path}")
    # Return unique alerts
    return list(set(alerts))

# --- This is the NEW function you are adding for Day 2 ---
def process_frame(frame):
    """
    Processes a single frame from a video or webcam feed.
    Returns the annotated frame and a list of detected object names.
    """
    # --- 1. Run the AI on the single frame ---
    results = model(frame)
    
    # --- 2. Get the frame with detection boxes drawn on it ---
    annotated_frame = results[0].plot()
    
    # --- 3. (Stretch Goal) Get a list of detected object names ---
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        detections.append(class_name)
        
    # --- 4. Return both the image and the list of detections ---
    return annotated_frame, detections

# This block is for your own testing if needed
if __name__ == '__main__':
    process_video_file('test_video.mp4', 'output.mp4')