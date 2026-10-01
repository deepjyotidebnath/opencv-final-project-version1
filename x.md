1. **Basic Object Detection on Video** — Run YOLOv8 on a video file frame-by-frame and draw bounding boxes; your first working detection pipeline.

2. **Object Tracking with ByteTrack** — Add persistent IDs across frames so the same object keeps the same track number as it moves, not just per-frame detections.

3. **Speed Estimation & Zone Counting** — Calculate each tracked object's pixel/second speed and count objects separately across two zones of the frame.

4. **FastAPI Backend with Live Streaming** — Wrap the tracker in a FastAPI service that processes video as a background job and streams annotated frames (MJPEG) to a client.

5. **Real-Time Multi-Object Tracking & Analytics Dashboard (Full App)** — Add rolling feature-window aggregation, XGBoost forecasting of next-window count/congestion, and a Streamlit dashboard with live video, metrics, and trend/forecast charts.