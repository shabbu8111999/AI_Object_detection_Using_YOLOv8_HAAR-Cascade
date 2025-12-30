import cv2
from config import FALLBACK_FPS

def get_video_properties(cap):
    """
    Safely extract video properties
    """

    # Get video properties safely
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read FPS safely
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fallback if FPS is invalid
    if fps <= 0 or fps > 120:
        fps = FALLBACK_FPS

    return width, height, fps

def create_writer(output_path, fps, width, height):
    """
    Create OpenCV VideoWriter
    """

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Return VideoWriter object
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))