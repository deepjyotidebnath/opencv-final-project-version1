# streamlit_app.py
# pip install streamlit ultralytics opencv-python
# run: streamlit run streamlit_app.py
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Video Detection", page_icon="🎯", layout="wide")
st.title("🎯 YOLOv8 Object Detection on Video")


@st.cache_resource
def load_model(name: str) -> YOLO:
    return YOLO(name)  # downloads weights on first run


with st.sidebar:
    st.header("Settings")
    weights = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.4, 0.05)
    preview_every = st.slider("Preview every N frames", 1, 10, 2)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded and st.button("Run detection", type="primary"):
    model = load_model(weights)

    # Save upload to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded.read())
        in_path = tmp_in.name
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Cannot open the uploaded video.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    progress = st.progress(0.0, text="Processing...")
    frame_slot = st.empty()
    counts: dict[str, int] = {}
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=conf, verbose=False)[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = model.names[int(box.cls[0])]
            counts[name] = counts.get(name, 0) + 1
            label = f"{name} {float(box.conf[0]):.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)

        if i % preview_every == 0:
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             caption=f"Frame {i + 1}/{total}", use_container_width=True)
        i += 1
        progress.progress(min(i / total, 1.0), text=f"Processing frame {i}/{total}")

    cap.release()
    writer.release()
    progress.empty()
    st.success(f"Done: {i} frames processed.")

    if counts:
        st.subheader("Detections (total across frames)")
        st.bar_chart(counts)

    with open(out_path, "rb") as f:
        st.download_button("Download annotated video", f, file_name="output.mp4",
                           mime="video/mp4")