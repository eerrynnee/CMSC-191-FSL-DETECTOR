# Filipino Sign Language Detector

CMSC 191 Computer Vision Final Project

Real-time Filipino Sign Language (FSL) detection using YOLOv11, OpenCV, and Streamlit.

## Features

- Live webcam detection with bounding boxes and gesture labels
- Haar cascade face exclusion to prevent false positives
- HSV skin segmentation and morphological filtering
- Confidence-weighted temporal smoothing to reduce flickering
- Clean dark-mode dashboard UI

## Setup

1. Create and activate a virtual environment:

   python -3.12 -m venv venv
   source venv/bin/activate # macOS/Linux
   venv\Scripts\activate # Windows

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run app.py

## Project Structure

- app.py — Streamlit UI and main application logic
- predict.py — Roboflow inference client
- preprocess.py — OpenCV CV pipeline
- drawing.py — Bounding box and label rendering
- smoothing.py — Temporal prediction smoother
- utils/ — Image conversion helpers
- assets/ — CSS styles

## CV Techniques

- Haar Cascade face detection (OpenCV built-in)
- HSV color space skin segmentation
- Morphological open/close operations
- Contour detection and bounding box extraction
- YOLOv11 object detection (Roboflow hosted)
- Confidence-weighted temporal smoothing
