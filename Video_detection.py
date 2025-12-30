import cv2
import time
from ultralytics import YOLO

# CONFIGURATION

# Path to video file
VIDEO_PATH = "data/video_data.mp4"

# Minimum confidence to show detection
CONF_THRESHOLD = 0.5

YOLO_OUTPUT_PATH = "yolo_output.mp4"
HAAR_OUTPUT_PATH = "haar_output.mp4"


# LOAD MODELS

def load_models():
  """
  Load YOLOv8 and HAAR Cascade model
  """

  # Loading YOLOv8 small models for fast and accuracte
  yolo_model = YOLO("yolov8n.pt")

  # Loading HAAR Cascade for Face Detection
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )

  return yolo_model, face_cascade


# DRAW YOLOv8 DETECTIONS

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


# MAIN FUNCTION

def main():
  # Open video file
  cap = cv2.VideoCapture(VIDEO_PATH)

  # Checking if video opened correctly
  if not cap.isOpened():
    print("Error: Video file not found")
    return
  
  # Get video properties for VideoWriter
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Creating video writers for YOLO and HAAR outputs
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  yolo_writer = cv2.VideoWriter(
    YOLO_OUTPUT_PATH, fourcc, fps, (width, height)
  )

  haar_writer = cv2.VideoWriter(
    HAAR_OUTPUT_PATH, fourcc, fps, (width, height)
  )

  # Loading both models
  yolo_model, face_cascade = load_models()

  # Used for FPS Calculation
  #prev_time = 0

  while True:
    # Reading video frame
    ret, frame = cap.read()
    if not ret:
      break

    # creating separate copies for YOLO and HAAR outputs
    yolo_frame = frame.copy()
    haar_frame = frame.copy()

    # Converting Frame to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # YOLOv8 Detection
    results = yolo_model(frame, verbose = False)
    draw_yolo_boxes(yolo_frame, results)

    # HAAR Detection
    draw_haar_boxes(haar_frame, gray, face_cascade)

    # Writing the frames to respective videos
    yolo_writer.write(yolo_frame)
    haar_writer.write(haar_frame)

    # Dispalying combined preview
    combined = cv2.hconcat([yolo_frame, haar_frame])
    cv2.imshow("YOLOv8 (Left) + HAAR (Right)", combined)

    # FPS Calculation
    #curr_time = time.time()
    #fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    #prev_time = curr_time

    # Displaying FPS on Screen
    #cv2.putText(
        #frame,
        #f"FPS: {int(fps)}",
        #(20, 40),
        #cv2.FONT_HERSHEY_SIMPLEX,
        #1,
        #(0, 0, 255),
        #2
    #)

    # Showing the Output window
    #cv2.imshow("YOLOv8 + HAAR Object Detection", frame)

    # Press ESC Key to exit
    if cv2.waitKey(1) & 0xFF == 27:
      break

  # Release Resources
  cap.release()
  yolo_writer.release()
  haar_writer.release()
  cv2.destroyAllWindows()

  print("Saved:")
  print(" - YOLO output:", YOLO_OUTPUT_PATH)
  print(" - HAAR output:", HAAR_OUTPUT_PATH)

# Runing the Program
main()