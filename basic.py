# streamlit_app.py
# run locally: streamlit run streamlit_app.py
import os
import shutil
import subprocess
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

MAX_SIDE = 640      # downscale frames so the longest side is at most this
MAX_FRAMES = 600    # safety cap for the free Streamlit Cloud tier

st.set_page_config(page_title="YOLOv8 Video Detection", page_icon="🎯", layout="wide")
st.title("🎯 YOLOv8 Object Detection on Video")


@st.cache_resource
def load_model(name: str) -> YOLO:
    return YOLO(name)  # downloads weights on first run


def reencode_h264(src: str) -> str | None:
    """Re-encode to H.264 so browsers can play it. Returns new path or None."""
    if shutil.which("ffmpeg") is None:
        return None
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", dst],
            check=True, capture_output=True,
        )
        return dst
    except subprocess.CalledProcessError:
        return None


def process_video(in_path: str, out_path: str, model: YOLO, conf: float,
                  preview_every: int):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return None, "Cannot open the uploaded video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, MAX_FRAMES)

    scale = min(1.0, MAX_SIDE / max(w, h))
    out_w, out_h = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    progress = st.progress(0.0, text="Processing...")
    frame_slot = st.empty()
    counts: dict[str, int] = {}
    i = 0

    try:
        while i < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break

            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            result = model(frame, conf=conf, verbose=False)[0]

            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                counts[name] = counts.get(name, 0) + 1

            annotated = result.plot()  # BGR frame with boxes + labels
            writer.write(annotated)

            if i % preview_every == 0:
                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 caption=f"Frame {i + 1}/{total}")
            i += 1
            progress.progress(min(i / total, 1.0),
                              text=f"Processing frame {i}/{total}")
    finally:
        cap.release()
        writer.release()
        progress.empty()
        frame_slot.empty()

    return {"frames": i, "counts": counts, "truncated": i >= MAX_FRAMES}, None


with st.sidebar:
    st.header("Settings")
    weights = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.4, 0.05)
    preview_every = st.slider("Preview every N frames", 1, 10, 2)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded and st.button("Run detection", type="primary"):
    model = load_model(weights)
    in_path = out_path = final_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded.read())
            in_path = tmp_in.name
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        info, err = process_video(in_path, out_path, model, conf, preview_every)
        if err:
            st.error(err)
        else:
            final_path = reencode_h264(out_path) or out_path
            with open(final_path, "rb") as f:
                info["video"] = f.read()
            info["h264"] = final_path != out_path
            st.session_state["result"] = info
    finally:
        for p in {in_path, out_path, final_path}:
            if p and os.path.exists(p):
                os.remove(p)

# Results live in session_state so the download button doesn't wipe them on rerun
res = st.session_state.get("result")
if res:
    st.success(f"Done: {res['frames']} frames processed.")
    if res["truncated"]:
        st.warning(f"Stopped at the {MAX_FRAMES}-frame limit.")

    if res["h264"]:
        st.video(res["video"])

    if res["counts"]:
        st.subheader("Detections (total across frames)")
        st.bar_chart(res["counts"])
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")

    st.download_button("Download annotated video", res["video"],
                       file_name="output.mp4", mime="video/mp4")