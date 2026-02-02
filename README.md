# 🪖 AI-Based Real-Time Helmet Detection System

## 📌 Overview
This project is a real-time computer vision system that detects whether a person is wearing a helmet using a live camera feed.  
It uses deep learning to analyze video frames and classify helmet vs no-helmet in real time.  
The system can be used in **construction sites, factories, labs, warehouses, and safety zones** to improve safety compliance.

## 🚀 Features
- 🎥 Real-time helmet detection using camera  
- 🧠 AI/Deep Learning based detection  
- ⚡ Fast and accurate results (YOLO-based)  
- 📦 Works with webcam / CCTV / IP camera  
- 🖼️ Live bounding boxes with confidence score  
- 📸 Option to save screenshots of violations (optional)  
- 🔔 Can be extended to alert system (buzzer / notification)  

## 🧠 Tech Stack
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- PyTorch  
- NumPy  

## 🗂️ Project Structure
helmet-detection/
├── models/ # Trained YOLO model (.pt file)
├── data/ # Dataset (if included)
├── src/
│ ├── main.py # Run real-time detection
│ ├── detector.py # Detection logic
│ └── utils.py # Helper functions
├── requirements.txt # Project dependencies
└── README.md

<img width="1919" height="941" alt="Screenshot 2026-02-03 015924" src="https://github.com/user-attachments/assets/c0379610-dd8f-42df-86e2-c49da4067909" />

## 🔮 Future Scope
- 🔍 **Live penalty system** by integrating:
  - Face detection & recognition for rider identification  
  - Number plate detection (ANPR) for vehicle identification  
- 🗄️ Automatic storage of violation data in a database  
- ☁️ Cloud dashboard for monitoring safety violations  
- 📱 Real-time alerts to admin / supervisor  
- 🤖 Multi-PPE detection (helmet, mask, gloves, safety shoes)  
- ⚙️ Deployment on edge devices (Raspberry Pi / Jetson Nano)  
- 🔒 Privacy features like automatic face blurring for non-violators
- 

