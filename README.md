# AI Object Detection using YOLOv8 and Haar Cascade
## Overview

This project focuses on object detection in video data using two different approaches: YOLOv8, a modern deep learning–based model, and Haar Cascade, a traditional computer vision technique.

The purpose of this project is to understand how object detection has evolved over time and to compare the accuracy, speed, and practicality of both methods in real-world scenarios.

The application processes a video and generates detection results separately for each approach, allowing a clear comparison between traditional and AI-driven solutions.

---

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Haar Cascade Classifier
- NumPy

---

## Detection Approaches Explained
### **YOLOv8 (Deep Learning)**

YOLOv8 is a state-of-the-art object detection model that uses deep neural networks to detect multiple objects in a single frame. It performs detection in real time and is widely used in production systems such as surveillance, autonomous vehicles, and smart cameras. This approach provides high accuracy and robustness across different object types and environments.

### **Haar Cascade (Traditional Computer Vision)**

Haar Cascade is a classical machine learning technique that relies on handcrafted features. It is mainly used for face detection and works efficiently on CPU-only systems. While it is lightweight and fast, it is limited in accuracy and flexibility compared to deep learning models.

---

## Key Features
- Video-based object detection
- Comparison of deep learning vs traditional methods
- Real-time frame processing
- Separate output results for each model
- Performance and accuracy observation

---

## Comparison Summary

| Aspect | YOLOv8 | Haar Cascade |
|------|--------|--------------|
| Approach | Deep Learning | Traditional ML |
| Accuracy | High | Moderate |
| Object Detection | Multiple classes | Limited |
| Scalability | High | Low |
| Resource Usage | Moderate | Low |

---

## Challenges Faced
- Handling video frame rates correctly
- Maintaining smooth output video duration
- Tuning detection confidence thresholds
- Understanding limitations of traditional models
- Comparing performance fairly between two approaches

---

## Learning Outcome

Through this project, I gained practical understanding of:
- How modern object detection models work internally
- Differences between classical and deep learning approaches
- Real-time video processing using OpenCV
- Performance considerations in AI-based systems

---

## Future Improvements
- Display FPS and detection confidence on video
- Use custom-trained YOLOv8 models
- Add object tracking functionality
- Integrate a simple UI for visualization
- Extend detection to live camera streams

---

## Conclusion

This project highlights the transition from traditional computer vision techniques to modern AI-powered object detection. It serves as a strong foundational project for anyone learning Computer Vision, Artificial Intelligence, and real-time video analysis, especially for beginners and freshers.