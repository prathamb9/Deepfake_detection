import streamlit as st
import requests
import tempfile
import cv2
import os

# PAGE CONFIG
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="wide")

st.title("Deepfake Detection System")
st.markdown("### Upload a video to check if it's **Real** or **Fake**")

uploaded_file = st.file_uploader("Choose a video file ", type=["mp4", "mov", "avi", "webm"])

if uploaded_file is not None:
    # Save & Inspect
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    video_path = tfile.name

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        st.info(f"📹 **Video Details:** {total_frames} Frames | {duration:.2f} Seconds | {fps:.2f} FPS")
        
        # Display Video (Centered)
        col1, col2, col3 = st.columns([3, 2, 3]) 
        with col2:
            st.video(uploaded_file)
        
        if total_frames < 10:
            st.error("⚠️ Video is too short! Needs at least 10 frames.")
        else:
            st.write("---")
            st.subheader("⚙️ Analysis Settings")
            
            # Slider
            frames_to_check = st.slider(
                "Select number of frames to analyze:",
                min_value=10, max_value=total_frames, value=min(20, total_frames), step=1
            )

            # Analyze Button
            b1, b2, b3 = st.columns([1, 1, 1])
            with b2:
                analyze_button = st.button("Analyze Video", use_container_width=True)

            if analyze_button:
                with st.spinner(f"Processing {frames_to_check} frames..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"frames_to_check": frames_to_check}
                    
                    try:
                        # Connect to clean endpoint
                        response = requests.post("http://127.0.0.1:8001/predict", files=files, data=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if result.get("status") == "success":
                                label = result["label"]
                                conf = result["confidence"]
                                raw_score = result["raw_score"]
                                
                                st.markdown("---")
                                st.markdown("##Final Verdict")
                                
                                res_col1, res_col2 = st.columns(2)
                                
                                with res_col1:
                                    if label == "FAKE":
                                        st.error(f"#FAKE DETECTED")
                                    else:
                                        st.success(f"#REAL VIDEO")
                                
                                with res_col2:
                                    st.metric("Confidence Score", conf)
                                    # Progress bar: If Fake, show score. If Real, show (1-score).
                                    prog_val = raw_score if label == "FAKE" else (1.0 - raw_score)
                                    st.progress(prog_val)
                                    
                            else:
                                st.error(f"Error: {result.get('message')}")
                        else:
                            st.error("Failed to connect to Backend.")
                            
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except PermissionError:
                pass