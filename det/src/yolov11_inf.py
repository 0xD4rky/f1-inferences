import numpy as np
from ultralytics import YOLO
import cv2

video_path = r'det/data/football.mp4'
model = YOLO("yolov11n.pt")  # Corrected model name

# Perform object detection on the video
results = model(video_path)

# Display the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow("YOLOv11 Inference", im_array)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cv2.destroyAllWindows()
