# 📌 Face Detection System

## 📖 Overview

This project implements a real-time face detection system using OpenCV and Python.
It detects human faces in images, videos, and webcam streams and highlights them using bounding boxes.

---

## 🎯 Objectives

* To detect faces in real-time using computer vision techniques
* To understand and apply pre-trained machine learning models
* To build a practical AI application based on AI & ML concepts

---

## 🧠 Concepts Used

* Computer Vision
* Machine Learning (Pre-trained Models)
* Haar Cascade Classifier
* Image Processing (Grayscale conversion)

---

## ⚙️ Features

* Real-time face detection using webcam
* Face detection from images
* Face detection from video files
* Displays number of faces detected
* FPS (Frames Per Second) display
* Error handling for invalid inputs

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy

---

## 📁 Project Structure

face-detection-project/
│── main.py
│── requirements.txt
│── README.md

├── models/
│   └── haarcascade_frontalface_default.xml

├── screenshots/
│   ├── menu.png
│   ├── webcam_detection.png
│   ├── image_detection.png
│   ├── multiple_faces.png
│   └── video_detection.png

---

## ▶️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/tiwarirajat-1501/face-detection-project.git
cd face-detection-project

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the project

python main.py

---

## 🧪 How It Works

1. The input (image/video/webcam) is captured
2. The frame is converted into grayscale
3. Haar Cascade classifier is applied
4. Faces are detected and marked with rectangles
5. Output is displayed in real-time

---

## 📸 Output Screenshots

(Add your screenshots in the screenshots/ folder and display here)

---

## 📊 Results

* Successfully detects faces in real-time
* Works on multiple faces simultaneously
* Efficient performance with good FPS

---

## 🚀 Applications

* Security and surveillance systems
* Attendance systems
* Face-based authentication
* Human-computer interaction

---

## 🔮 Future Enhancements

* Face recognition (identify person)
* Mask detection system
* Emotion detection
* GUI using Streamlit or Tkinter

---

## 🧠 Viva Questions (Preparation)

* What is face detection vs face recognition?
* Why convert image to grayscale?
* What is Haar Cascade?
* What are parameters like scaleFactor and minNeighbors?

---

## 👨‍💻 Author

Rajat Tiwari
B.Tech CSE (AI & ML)

---

## 📜 License

This project is for educational purposes.
