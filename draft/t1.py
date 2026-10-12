"""
Streamlit app: Object tracking with ByteTrack (persistent IDs across frames).

Install:
    pip install streamlit ultralytics supervision opencv-python pandas imageio-ffmpeg

Run:
    streamlit run app.py
"""

import os
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import supervision as sv
from ultralytics import YOLO

st.set_page_config(page_title="ByteTrack Object Tracker", page_icon="🎯", layout="wide")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detector...")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def to_h264(src: str, dst: str) -> bool:
    """Re-encode to H.264 so the browser can play it. Returns True on success."""
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
            [
                exe, "-y", "-i", src,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-movflags", "+faststart", dst,
            ],
            capture_output=True,
        )
        return r.returncode == 0 and os.path.exists(dst)
    except Exception:
        return False


def run_tracking(
    video_path: str,
    model: YOLO,
    class_ids,
    det_conf: float,
    activation: float,
    buffer: int,
    match: float,
    iou: float,
    imgsz: int,
    trail: int,
    line_y: float | None,
    max_frames: int,
    preview_every: int,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames > 0:
        total = min(total, max_frames) if total else max_frames

    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = sv.ByteTrack(
        track_activation_threshold=activation,
        lost_track_buffer=buffer,
        minimum_matching_threshold=match,
        frame_rate=int(round(fps)),
    )
    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=trail) if trail > 0 else None

    line_zone = line_ann = None
    if line_y is not None:
        y = int(height * line_y)
        line_zone = sv.LineZone(start=sv.Point(0, y), end=sv.Point(width, y))
        line_ann = sv.LineZoneAnnotator(thickness=2, text_scale=0.8)

    names = model.names
    tracks: dict[int, dict] = {}
    active_per_frame: list[int] = []

    progress = st.progress(0.0, text="Starting...")
    preview = st.empty()
    t0 = time.time()
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames > 0 and idx >= max_frames):
                break
            idx += 1

            result = model.predict(
                frame, conf=det_conf, iou=iou, imgsz=imgsz,
                classes=class_ids, verbose=False,
            )[0]
            det = sv.Detections.from_ultralytics(result)
            det = tracker.update_with_detections(det)
            if det.tracker_id is None:
                det.tracker_id = np.array([], dtype=int)

            for tid, cid, conf in zip(det.tracker_id, det.class_id, det.confidence):
                tid = int(tid)
                t = tracks.setdefault(
                    tid,
                    {"class": names[int(cid)], "first": idx, "last": idx, "frames": 0, "conf": 0.0},
                )
                t["last"] = idx
                t["frames"] += 1
                t["conf"] += float(conf)
            active_per_frame.append(len(det))

            labels = [
                f"#{tid} {names[int(cid)]} {conf:.2f}"
                for tid, cid, conf in zip(det.tracker_id, det.class_id, det.confidence)
            ]
            out = frame.copy()
            if trace_ann is not None:
                out = trace_ann.annotate(out, det)
            out = box_ann.annotate(out, det)
            out = label_ann.annotate(out, det, labels=labels)
            if line_zone is not None:
                line_zone.trigger(det)
                out = line_ann.annotate(out, line_counter=line_zone)
            cv2.putText(
                out, f"Active: {len(det)}  Unique IDs: {len(tracks)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            writer.write(out)

            if idx % preview_every == 0:
                preview.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB",
                              caption=f"Frame {idx}", use_container_width=True)
            if total:
                elapsed = time.time() - t0
                progress.progress(
                    min(idx / total, 1.0),
                    text=f"Frame {idx}/{total}  •  {idx / max(elapsed, 1e-6):.1f} FPS",
                )
    finally:
        cap.release()
        writer.release()

    preview.empty()
    progress.empty()

    # Browser-playable version
    final_out = raw_out.replace(".mp4", "_h264.mp4")
    playable = to_h264(raw_out, final_out)
    video_path_final = final_out if playable else raw_out
    with open(video_path_final, "rb") as f:
        video_bytes = f.read()
    for p in {raw_out, final_out}:
        if os.path.exists(p):
            os.remove(p)

    rows = [
        {
            "ID": tid,
            "Class": t["class"],
            "First frame": t["first"],
            "Last frame": t["last"],
            "Frames seen": t["frames"],
            "Avg conf": round(t["conf"] / t["frames"], 3),
            "On screen (s)": round(t["frames"] / fps, 2),
        }
        for tid, t in sorted(tracks.items())
    ]
    df = pd.DataFrame(rows)

    return {
        "video": video_bytes,
        "playable": playable,
        "df": df,
        "active": active_per_frame,
        "frames": idx,
        "fps": fps,
        "elapsed": time.time() - t0,
        "in": line_zone.in_count if line_zone else None,
        "out": line_zone.out_count if line_zone else None,
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🎯 Object Tracking with ByteTrack")
st.caption("YOLO detections + ByteTrack = persistent IDs that follow each object across frames.")

with st.sidebar:
    st.header("Settings")

    weights = st.selectbox(
        "Detector", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"], index=0,
        help="Larger models are more accurate but slower.",
    )
    model = load_model(weights)

    all_names = list(model.names.values())
    picked = st.multiselect("Classes to track", all_names, default=[],
                            help="Leave empty to track every class.")
    class_ids = [k for k, v in model.names.items() if v in picked] or None

    st.subheader("ByteTrack")
    det_conf = st.slider("Detector confidence", 0.05, 0.60, 0.10, 0.05,
                         help="Keep low: ByteTrack uses low-score boxes to keep tracks alive.")
    activation = st.slider("Track activation threshold", 0.10, 0.90, 0.50, 0.05,
                           help="Minimum score for a detection to start a new track.")
    buffer = st.slider("Lost-track buffer (frames)", 5, 150, 30, 5,
                       help="How long a track survives while occluded/undetected.")
    match = st.slider("IoU matching threshold", 0.30, 0.95, 0.80, 0.05)

    st.subheader("Display / speed")
    imgsz = st.select_slider("Inference size", [320, 480, 640, 800, 960, 1280], value=640)
    iou = st.slider("NMS IoU", 0.30, 0.95, 0.70, 0.05)
    trail = st.slider("Trail length (frames, 0 = off)", 0, 120, 40, 5)
    use_line = st.checkbox("Line-crossing counter")
    line_y = st.slider("Line position (fraction of height)", 0.05, 0.95, 0.50, 0.05) if use_line else None
    max_frames = st.number_input("Max frames (0 = whole video)", min_value=0, value=0, step=100)
    preview_every = st.slider("Live preview every N frames", 1, 20, 3)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    if st.session_state.get("upload_name") != uploaded.name:
        # New file -> reset previous results and store to disk
        st.session_state.pop("result", None)
        old = st.session_state.get("video_path")
        if old and os.path.exists(old):
            os.remove(old)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state["video_path"] = tmp.name
        st.session_state["upload_name"] = uploaded.name

    left, right = st.columns(2)
    with left:
        st.subheader("Input")
        st.video(st.session_state["video_path"])

    run = st.button("▶ Run tracking", type="primary")
    if run:
        with right:
            st.subheader("Processing")
            try:
                st.session_state["result"] = run_tracking(
                    st.session_state["video_path"], model, class_ids, det_conf, activation,
                    buffer, match, iou, imgsz, trail, line_y, int(max_frames), preview_every,
                )
            except Exception as e:
                st.error(f"Tracking failed: {e}")

    res = st.session_state.get("result")
    if res:
        with right:
            st.subheader("Tracked output")
            if res["playable"]:
                st.video(res["video"])
            else:
                st.warning(
                    "Couldn't re-encode for browser playback (ffmpeg missing). "
                    "Install `imageio-ffmpeg` or download the file below."
                )

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique objects", len(res["df"]))
        c2.metric("Frames processed", res["frames"])
        c3.metric("Avg FPS", f"{res['frames'] / max(res['elapsed'], 1e-6):.1f}")
        c4.metric("Peak concurrent", max(res["active"]) if res["active"] else 0)
        if res["in"] is not None:
            l1, l2 = st.columns(2)
            l1.metric("Line crossings: in", res["in"])
            l2.metric("Line crossings: out", res["out"])

        tab1, tab2 = st.tabs(["Track table", "Objects on screen over time"])
        with tab1:
            if not res["df"].empty:
                st.dataframe(res["df"], use_container_width=True, hide_index=True)
                st.bar_chart(res["df"]["Class"].value_counts(), y_label="unique objects")
            else:
                st.info("No confirmed tracks. Try lowering the activation threshold.")
        with tab2:
            st.line_chart(pd.Series(res["active"], name="active tracks"))

        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download video", res["video"], "tracked_output.mp4",
                           "video/mp4", use_container_width=True)
        d2.download_button("⬇ Download tracks (CSV)", res["df"].to_csv(index=False),
                           "tracks.csv", "text/csv", use_container_width=True)
else:
    st.info("Upload a video to begin.")