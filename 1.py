"""
Project 1: Basic Object Detection on Video (Streamlit version)
----------------------------------------------------------------
Upload a video, run YOLOv8 on it frame by frame, draw bounding boxes + labels,
and preview / download the annotated result.

Requires: pip install streamlit ultralytics opencv-python-headless
Usage:    streamlit run detect_video_streamlit.py
"""

import os
import tempfile
import time

import cv2
import streamlit as st
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Cached model loader so the weights aren't reloaded on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


def detect_video(
    input_path: str,
    output_path: str,
    model: YOLO,
    conf: float,
    progress_bar,
    status_text,
    frame_placeholder,
    preview_every: int = 5,
):
    """Run detection over every frame of input_path, write annotated frames
    to output_path, and update Streamlit UI elements as it goes."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, verbose=False)[0]
        annotated_frame = results.plot()

        writer.write(annotated_frame)
        frame_num += 1

        # Update progress + a live preview every few frames (keeps UI responsive)
        if total_frames:
            progress_bar.progress(min(frame_num / total_frames, 1.0))
        status_text.text(f"Processed frame {frame_num}/{total_frames or '?'}")

        if frame_num % preview_every == 0:
            frame_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_num}",
                use_container_width=True,
            )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    return frame_num, elapsed


def main():
    st.set_page_config(page_title="YOLOv8 Video Object Detection", layout="centered")
    st.title("🎥 YOLOv8 Video Object Detection")
    st.write(
        "Upload a video, and this app will run YOLOv8 object detection on every "
        "frame, drawing bounding boxes and labels, then let you preview and "
        "download the annotated result."
    )

    # --- Sidebar controls -------------------------------------------------
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Model",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Nano (n) is fastest; extra-large (x) is most accurate but slowest.",
    )
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

    # --- File upload --------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is None:
        st.info("Upload a video file to get started.")
        return

    st.video(uploaded_file)

    if st.button("Run Detection", type="primary"):
        # Save uploaded file to a temp path so OpenCV can open it
        in_suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix) as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        output_path = os.path.join(
            tempfile.gettempdir(), f"annotated_{os.path.basename(input_path)}.mp4"
        )

        with st.spinner(f"Loading {model_name}..."):
            model = load_model(model_name)

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        frame_placeholder = st.empty()

        try:
            frame_count, elapsed = detect_video(
                input_path,
                output_path,
                model,
                conf,
                progress_bar,
                status_text,
                frame_placeholder,
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return
        finally:
            os.remove(input_path)

        st.success(
            f"Done! Processed {frame_count} frames in {elapsed:.1f}s "
            f"({frame_count / elapsed:.1f} FPS)."
        )

        st.subheader("Annotated Video")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download annotated video",
                data=f,
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()