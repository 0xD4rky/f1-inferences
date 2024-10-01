import numpy as np
from ultralytics import YOLO
import torch

video_path = r'det/data/football.mp4'
model = YOLO("yolov8s.pt")

# Use tracking during prediction
results = model.track(source=video_path, device='cuda', show=True)

# Process the results (optional)
for r in results:
    print(f"Frame {r.frame_number}:")
    print(f"Detections: {r.boxes.data}")
    print(f"Tracks: {r.tracks}")