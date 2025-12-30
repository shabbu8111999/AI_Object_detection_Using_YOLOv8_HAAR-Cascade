# models.py

import cv2
from ultralytics import YOLO

def load_models():
    """
    Load YOLOv8 and HAAR Cascade models
    """

    # Loading YOLOv8 small models for fast and accuracte
    yolo_model = YOLO("yolov8n.pt")

    # Loading HAAR Cascade for Face Detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return yolo_model, face_cascade
