import streamlit as st
import cv2
import tempfile
from ai_core import process_video_file, process_frame

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Project AEGIS")

# --- Main Title ---
st.title("🛡 Project AEGIS: Automated Event Guardian & Intelligence System")
st.markdown("---")

# --- Sidebar Navigation ---
st.sidebar.header("Controls")
app_mode = st.sidebar.selectbox("Choose the App Mode",
    ["About", "Analyze Video File", "Live Feed Analysis"]
)

# --- About Page ---
if app_mode == "About":
    st.header("Mission")
    st.markdown("""
        This application uses Artificial Intelligence to automatically analyze video footage for the National Security Guard. 
        It is designed to detect potential threats in real-time from various sources like surveillance cameras, drones, and body cams, 
        thereby enhancing operational awareness and safety.
    """)
    st.markdown("---")
    st.subheader("Technology Stack")
    st.markdown("""
        - *Python:* The core programming language.
        - *Streamlit:* For creating the interactive web application.
        - *OpenCV:* For video processing and handling.
        - *YOLOv8 (Ultralytics):* The state-of-the-art AI model for real-time object detection.
    """)

# --- Analyze Video File Page ---
elif app_mode == "Analyze Video File":
    st.header("Analyze a Pre-recorded Video")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

        st.video(temp_video_path)
        
        if st.button("Analyze Video"):
            with st.spinner("Analyzing... This may take a few moments depending on the video length."):
                output_video_path = "output_video.mp4"
                
                # Call the AI function from ai_core.py
                alerts = process_video_file(temp_video_path, output_video_path)
            
            st.success("Analysis Complete!")
            
            # Display the processed video
            st.video(output_video_path)
            
            # Display any alerts that were generated
            if alerts:
                st.header("Detected Events:")
                st.table(alerts)
            else:
                st.info("No specific events were detected in this video.")

# --- Live Feed Analysis Page ---
elif app_mode == "Live Feed Analysis":
    st.header("Analyze Live Webcam Feed")
    st.warning("This feature will request access to your webcam. Ensure you have given your browser permission.")

    # Use session state to manage the run state of the webcam
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    if st.button('Start Live Feed'):
        st.session_state.run_webcam = True

    if st.button('Stop Live Feed'):
        st.session_state.run_webcam = False
        st.info("Live feed stopped.")

    if st.session_state.run_webcam:
        st.info("Webcam feed is running... (Click 'Stop Live Feed' to end)")
        
        # Placeholder for the video frames
        frame_placeholder = st.empty()
        
        # Open the webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            while cap.isOpened() and st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.write("The video stream has ended.")
                    break
                
                # Process the frame using the AI function from ai_core.py
                # This assumes process_frame returns the annotated_frame and a list/string of detections
                annotated_frame, detections = process_frame(frame)

                # Display the processed frame
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                
                # You can optionally display the list of detections
                # st.write(f"Detections: {', '.join(detections) if detections else 'None'}")

            # Release the webcam when the loop is broken
            cap.release()