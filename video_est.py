import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Define body parts and pose pairs
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# Load TensorFlow model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Streamlit App Title
st.title("Video Pose Detection with OpenCV")

# File uploader for video input
video_file = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

# Threshold slider for key point confidence
threshold = st.slider("Confidence Threshold for Key Points", min_value=0, max_value=100, value=50, step=5) / 100

# Pose estimation function
def process_video(video_path, confidence):
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frameHeight, frameWidth, _ = frame.shape
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = int((frameWidth * point[0]) / out.shape[3])
            y = int((frameHeight * point[1]) / out.shape[2])
            points.append((x, y) if conf > confidence else None)

        for pair in POSE_PAIRS:
            partFrom, partTo = pair
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
                cv2.circle(frame, points[idFrom], 4, (0, 0, 255), -1)
                cv2.circle(frame, points[idTo], 4, (0, 0, 255), -1)

        # Append processed frame to the list
        output_frames.append(frame)

    cap.release()
    return output_frames

# Process video if uploaded
if video_file:
    # Save uploaded video to temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Display uploaded video
    st.text("Original Video:")
    st.video("temp_video.mp4")

    # Process video
    st.text("Processing Video...")
    frames = process_video("temp_video.mp4", threshold)

    # Create a temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
        output_path = temp_output.name

        # Convert frames to a video file in the temporary file
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

        # Display the processed video (without showing the path)
        st.text("Processed Video with Pose Detection:")
        st.video(output_path)  # This will play the video

    st.success("Video Processing Complete!")
