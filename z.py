import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from speed_zones import ZONE_COLORS, SpeedEstimator, ZoneCounter, draw_hud, draw_zones, zone_of

st.set_page_config(page_title="Speed & Zone Counting", page_icon="🚦", layout="wide")

COCO_CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "bus": 5, "truck": 7, "dog": 16, "horse": 17,
}


@st.cache_resource
def get_model(name):
    return YOLO(name)


def build_zones(width, height, orientation, split_pct, custom_json):
    if custom_json.strip():
        raw = json.loads(custom_json)
        if len(raw) != 2:
            raise ValueError("Custom zones must define exactly two polygons")
        return {k: np.array(v, np.int32) for k, v in raw.items()}
    if orientation == "Vertical split (left | right)":
        s = int(width * split_pct / 100)
        return {
            "A": np.array([[0, 0], [s, 0], [s, height], [0, height]], np.int32),
            "B": np.array([[s, 0], [width, 0], [width, height], [s, height]], np.int32),
        }
    s = int(height * split_pct / 100)
    return {
        "A": np.array([[0, 0], [width, 0], [width, s], [0, s]], np.int32),
        "B": np.array([[0, s], [width, s], [width, height], [0, height]], np.int32),
    }


def to_browser_mp4(src):
    """Re-encode mp4v -> H.264 so st.video can play it (needs ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", dst]
    )
    return dst if r.returncode == 0 and os.path.exists(dst) else src


def process_video(path, model, zones, conf, classes, max_seconds, preview, progress, status):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    limit = min(total, int(max_seconds * fps)) if max_seconds else total

    names = list(zones)
    estimator = SpeedEstimator(fps)
    counter = ZoneCounter(names)
    out_raw = os.path.join(tempfile.mkdtemp(), "annotated.mp4")
    writer = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rows, occupancy = [], []
    frame_idx = 0
    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        counter.begin_frame()
        frame = draw_zones(frame, zones)

        res = model.track(
            frame, persist=frame_idx > 1, tracker="bytetrack.yaml",
            conf=conf, classes=classes, verbose=False,
        )[0]

        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()
            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clss):
                point = ((x1 + x2) / 2, y2)
                speed = estimator.update(tid, frame_idx, point)
                zone = zone_of(point, zones)
                counter.add(zone, tid)

                color = ZONE_COLORS.get(zone, (200, 200, 200))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"#{tid} {res.names[c]} {speed:.0f}px/s",
                            (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                rows.append({
                    "frame": frame_idx, "time_s": round(frame_idx / fps, 3), "track_id": tid,
                    "class": res.names[c], "zone": zone or "", "x": round(point[0], 1),
                    "y": round(point[1], 1), "speed_px_s": round(speed, 1),
                })

        occupancy.append({"time_s": round(frame_idx / fps, 3),
                          **{f"Zone {z}": len(counter.current[z]) for z in names}})
        draw_hud(frame, counter, names)
        writer.write(frame)

        if frame_idx % 5 == 0:
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            progress.progress(min(frame_idx / limit, 1.0))
            status.caption(f"Frame {frame_idx}/{limit}")

    cap.release()
    writer.release()
    progress.progress(1.0)

    return {
        "video": to_browser_mp4(out_raw),
        "df": pd.DataFrame(rows),
        "occupancy": pd.DataFrame(occupancy),
        "totals": {z: len(counter.total[z]) for z in names},
        "fps": fps,
    }


# ------------------------------- UI -------------------------------
st.title("🚦 Speed Estimation & Zone Counting")
st.caption("Track objects, measure pixel/second speed, and count them separately in two zones.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    picked = st.multiselect("Classes to track", list(COCO_CLASSES), default=["car", "motorcycle", "bus", "truck"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
    max_seconds = st.number_input("Process first N seconds (0 = all)", 0, 3600, 20)

    st.subheader("Zones")
    orientation = st.radio("Layout", ["Vertical split (left | right)", "Horizontal split (top / bottom)"])
    split_pct = st.slider("Split position (%)", 10, 90, 50)
    custom_json = st.text_area(
        "Custom polygons (optional JSON)",
        placeholder='{"A": [[0,0],[640,0],[640,720],[0,720]], "B": [[640,0],[1280,0],[1280,720],[640,720]]}',
        height=110,
    )

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tmp.write(uploaded.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    cap.release()

    try:
        zones = build_zones(W, H, orientation, split_pct, custom_json)
    except Exception as e:
        st.error(f"Invalid zones: {e}")
        st.stop()

    if ok:
        st.subheader("Zone preview")
        st.image(cv2.cvtColor(draw_zones(first, zones), cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("▶ Run analysis", type="primary"):
        st.session_state.pop("result", None)
        progress, status, preview = st.progress(0.0), st.empty(), st.empty()
        with st.spinner("Processing..."):
            st.session_state["result"] = process_video(
                tmp.name, get_model(model_name), zones, conf,
                [COCO_CLASSES[c] for c in picked] or None,
                max_seconds, preview, progress, status,
            )
        preview.empty()
        status.empty()

result = st.session_state.get("result")
if result:
    df, occ, totals = result["df"], result["occupancy"], result["totals"]

    st.subheader("Results")
    cols = st.columns(len(totals) + 2)
    for col, (z, n) in zip(cols, totals.items()):
        col.metric(f"Zone {z} unique objects", n)
    cols[-2].metric("Tracked objects", df["track_id"].nunique() if not df.empty else 0)
    cols[-1].metric("Avg speed (px/s)", f"{df['speed_px_s'].mean():.0f}" if not df.empty else "0")

    left, right = st.columns([3, 2])
    with left:
        st.video(result["video"])
        with open(result["video"], "rb") as f:
            st.download_button("⬇ Download annotated video", f, "annotated.mp4", "video/mp4")
    with right:
        st.markdown("**Zone occupancy over time**")
        st.line_chart(occ.set_index("time_s"))

    if not df.empty:
        st.markdown("**Per-object summary**")
        summary = (
            df.groupby("track_id")
            .agg(cls=("class", "first"),
                 frames=("frame", "count"),
                 avg_speed=("speed_px_s", "mean"),
                 max_speed=("speed_px_s", "max"),
                 zones=("zone", lambda s: ",".join(sorted({z for z in s if z}))))
            .round(1)
            .sort_values("max_speed", ascending=False)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("**Average speed by zone (px/s)**")
        zone_speed = df[df["zone"] != ""].groupby("zone")["speed_px_s"].mean().round(1)
        st.bar_chart(zone_speed)

        st.download_button("⬇ Download full CSV", df.to_csv(index=False), "speeds.csv", "text/csv")
    else:
        st.info("No objects were tracked. Try lowering confidence or selecting more classes.")