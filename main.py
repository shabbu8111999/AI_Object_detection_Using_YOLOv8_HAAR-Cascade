import cv2
import os

from config import VIDEO_PATH, YOLO_OUTPUT_PATH, HAAR_OUTPUT_PATH
from models import load_models
from utils.yolo_utils import draw_yolo_boxes
from utils.haar_utils import draw_haar_boxes
from utils.video_utils import create_writer

def main():
    # -------------------------------
    # Open video
    # -------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Video file not found")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -------------------------------
    # Force correct output settings
    # -------------------------------
    FPS = 25
    TARGET_DURATION = 15   # seconds
    TARGET_FRAMES = FPS * TARGET_DURATION

    print(f"Target output: {TARGET_DURATION} seconds")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(YOLO_OUTPUT_PATH), exist_ok=True)

    yolo_writer = create_writer(YOLO_OUTPUT_PATH, FPS, width, height)
    haar_writer = create_writer(HAAR_OUTPUT_PATH, FPS, width, height)

    yolo_model, face_cascade = load_models()

    frames_written = 0

    # -------------------------------
    # Read input frames
    # -------------------------------
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("Error: No frames read")
        return

    repeat_factor = TARGET_FRAMES // len(frames)
    repeat_factor = max(1, repeat_factor)

    print(f"Repeating each frame {repeat_factor} times")

    # -------------------------------
    # Write frames repeatedly
    # -------------------------------
    for frame in frames:
        for _ in range(repeat_factor):
            if frames_written >= TARGET_FRAMES:
                break

            yolo_frame = frame.copy()
            haar_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = yolo_model(frame, verbose=False)
            draw_yolo_boxes(yolo_frame, results)
            draw_haar_boxes(haar_frame, gray, face_cascade)

            yolo_writer.write(yolo_frame)
            haar_writer.write(haar_frame)

            frames_written += 1

    # -------------------------------
    # Cleanup
    # -------------------------------
    yolo_writer.release()
    haar_writer.release()
    cv2.destroyAllWindows()

    print("✅ OUTPUT SAVED SUCCESSFULLY")
    print(f"YOLO Output : {YOLO_OUTPUT_PATH}")
    print(f"HAAR Output : {HAAR_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
