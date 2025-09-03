import streamlit as st
import cv2
import numpy as np

###### Video Processing Utilities ######
def load_video(vid_path, output_path, bs=8):
    # Function to load video frames
    # Load video
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = None

    def load():
        nonlocal writer, fps
        while True:
            frames = []
            for i in range(bs):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if writer is None:
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame.shape[1], frame.shape[0]))
                frames.append(frame)
            if not ret:
                break
            else:
                yield frames

    def save(frame, fps):
        nonlocal writer
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),  # Position in top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return writer
    return load, save, cap, fps

def show_video(vid_path):
    with open(vid_path, 'rb') as f:
        st.video(f.read())  