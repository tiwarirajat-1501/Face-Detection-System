# Project Report: Face Detection System
**Course:** Fundamentals In Artificial Intelligence And Machine Learning

---

## 1. Executive Summary

This report presents the design and implementation of a multi-mode face detection system built using Python and OpenCV. The system operates across static (images), temporal (videos), and dynamic (webcam) environments. It leverages the classical Viola–Jones Haar Cascade framework to detect human faces efficiently. The project focuses on achieving a balance between computational efficiency and detection accuracy, enabling real-time performance on standard CPU hardware.

## 2. Introduction

Face detection is a foundational task in computer vision used in applications such as surveillance, authentication, and human–computer interaction. This project develops a lightweight and robust system that performs face detection without relying on heavy deep learning models, ensuring accessibility and efficiency.

### 2.1 Objectives

* Implement real-time face detection using webcam input.
* Enable detection in static images and pre-recorded videos.
* Optimize parameters to reduce false positives while maintaining performance (FPS).
* Design a modular and extensible system.

## 3. Methodology & Core Technologies

### 3.1 Viola–Jones (Haar Cascade)
The system uses a pre-trained Haar Cascade classifier (`haarcascade_frontalface_default.xml`). It computes Haar-like features over sliding windows and uses a cascade of classifiers to quickly reject non-face regions, focusing computation on promising areas.

### 3.2 Grayscale Conversion
Incoming frames are converted to grayscale using `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`. This reduces dimensionality and accelerates computation while preserving structural information necessary for detection.

### 3.3 Detection Pipeline
The pipeline includes input acquisition, preprocessing (grayscale), feature extraction, classification via the cascade, and rendering bounding boxes around detected faces.

## 4. System Implementation

### 4.1 Execution Modes
* **Webcam Mode:** Real-time capture using `cv2.VideoCapture(0)` with FPS monitoring.
* **Image Mode:** Single-pass detection using `cv2.imread()`.
* **Video Mode:** Frame-by-frame processing of pre-recorded videos.

### 4.2 Parameter Tuning
The `detectMultiScale` function is tuned with parameters such as `scaleFactor=1.2`, `minNeighbors=6`, and `minSize=(40,40)`. These control detection sensitivity, reduce false positives, and ignore noise from very small regions.

## 5. Results & Analysis

The system demonstrates stable real-time performance with consistent FPS on CPU hardware. It accurately detects multiple faces in varying conditions, while tuned parameters significantly reduce false detections.

## 6. Screenshots

*(Note: Insert your actual screenshot images here. Below are placeholders based on the report images.)*

* **Screenshot 1:** *VS Code interface showing the CLI execution and a popup window successfully detecting three faces in a sample image (`test_image.jpg`) with green bounding boxes.*
* **Screenshot 2:** *VS Code interface showing the CLI execution and a popup window successfully detecting four faces in another sample image (`test_image2.jpg`) with green bounding boxes.*

## 7. Advantages

* Lightweight and efficient (no GPU required).
* Real-time processing capability.
* Simple and modular implementation.
* Suitable for low-resource systems.

## 8. Limitations

* Sensitive to lighting variations.
* Lower accuracy compared to deep learning models.
* Difficulty with occlusions and extreme face angles.

## 9. Future Scope

* Integrate deep learning-based detectors.
* Add face recognition module.
* Develop GUI using Tkinter/Streamlit.
* Use multi-threading for higher FPS.
* Extend to emotion and mask detection.

## 10. Conclusion

This project successfully demonstrates a practical, efficient face detection system using classical computer vision techniques. It validates the effectiveness of Haar Cascade classifiers for real-time applications and provides a strong foundation for further enhancements.
