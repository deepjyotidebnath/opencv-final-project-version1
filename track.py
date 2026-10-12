import os
import shutil
import subprocess
import tempfile
import time

import cv2
import pandas as pd
import streamlit as st 
import supervision as sv
from ultralytics import YOLO

st.set_page_config(page_title="ByteTrack Object Tracker", page_icon="🎯", layout="wide")


@st.cache_resouce(show_spinner="Loading detector.....")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)

def to_h264(src:str, dst:str) ->bool:
    """re-encode to H.264 so the browser can playit. Returns True on success."""
    exe = None
    try:
        import imageio_ffmpeg
        
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        r = subprocess.run(
             
        )