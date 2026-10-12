import json
import os
import shutil
import subprocess
import tempfile
import pandas as pd
import streamlit as st 



import csv 
import json
import math 
from collections import defacultdict, deque 

import cv2 
import numpy as np 
from ultralytics import YOLO 

from speed_zones import ZONE_COLORS, SpeedEstimator,ZoneCounter, draw_hub, draw_zones, zone_of

st.set_page_config(page_title="Speed & Zone Counting", page_icon="🚦", layout="wide")













def load_zones(path, width, height):
    """two polygons . Default to left half (A) and right half(B) of the frame."""
    
    if path:
        with open(path) as f:
            raw = json.load(f)
            
        zones = {k: np.array(v, np.int32) for k, v in raw.items()}
        if len(zones) !=2:
            raise ValueError("Zones file must define exactly two zones")
        
        return zones 
    mid = width//2 
    return {
        "A":np.array([[0,0],[mid,height],[0,height]], np.int32),
        "B":np.array([[mid, 0],[width, 0],[width,height],[mid,height]],np.int32),
        
    }
    
            