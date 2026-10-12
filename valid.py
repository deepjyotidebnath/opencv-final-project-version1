from ultralytics import YOLO

model = YOLO("yolov8n.pt")
metrics = model.val(data="data.yml")
print(metrics.box.map, metrics.box.map50, metrics.box.mp, metrics.box.mr)