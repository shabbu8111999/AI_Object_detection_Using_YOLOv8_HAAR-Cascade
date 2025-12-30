# yolo_utils.py

import cv2
from config import CONF_THRESHOLD

def draw_yolo_boxes(frame, results):
    """
    Draw bounding boxes for YOLOv8 detections
    """

    # Loop through detected boxes
    for box in results[0].boxes:

        # Get confidence score
        conf = box.conf[0].item()
        if conf < CONF_THRESHOLD:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get detected class names
        class_id = int(box.cls[0])
        label = f"{results[0].names[class_id]}: {conf:.2f}"

        # Draw rectangle around object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
