import os
import tempfile
import time

import cv2
import streamlit as st 
from ultralytics import YOLO 

@st.cache_resource(show_spinner= False)
def load_model(model_name: str)-> YOLO:
    return YOLO(model_name)

def detect_video(
    input_path: str,
    output_path: str,
    model: YOLO,
    conf: float,
    progress_bar,
    status_text,
    frame_placeholder,
    preview_every: int=5,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video; {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0 
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num =0
    start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        results= model(frame, conf=conf , verbose= False)[0]
        annotated_frame = results.plot()
        
        writer.write(annotated_frame)
        frame_num += 1
        
        
        
        
        
        
        
        
        
        
        
    
    
def main():
    st.set_page_config(page_title="YOLOv8 Video object Detection", layout="centered")
    st.title("🎥 YOLOv8 Video Object Detection")