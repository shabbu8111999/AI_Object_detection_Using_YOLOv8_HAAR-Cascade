# haar_utils.py

import cv2

# DRAW HAAR DETECTIONS

def draw_haar_boxes(frame, gray, cascade):
  """
  Draw bounding boxes for HAAR Face Detections
  """

  # Detect faces from grayscale image
  faces = cascade.detectMultiScale(gray, 1.1, 5)

  # Loop through detected faces
  for (x, y, w, h) in faces:

    # Draw rectangle boxes frames
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )