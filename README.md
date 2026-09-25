# Real-Time Emotion Detection (CNN + OpenCV)

A computer vision project that detects faces from a live webcam feed and
classifies facial expressions into 6 emotions using a custom-trained CNN,
combined with OpenCV's Haar Cascade face detector.

## Problem Statement

Recognizing human emotions from facial expressions in real time has
applications in customer feedback analysis, mental health monitoring, and
human-computer interaction. This project builds an end-to-end pipeline:
detect faces in a live video stream, then classify each detected face's
emotion.

## Dataset

**FER2013** (Facial Expression Recognition) - ~35,000 grayscale 48x48 face
images labeled with 7 emotions. The `disgust` class was excluded (very few
samples), leaving 6 classes: angry, fear, happy, neutral, sad, surprise.

## Approach

1. **Face Detection** - OpenCV's Haar Cascade classifier (classical computer
   vision, built into OpenCV) locates faces in each webcam frame.
2. **Emotion Classification** - A custom 4-block CNN (Conv2D + BatchNorm +
   MaxPooling, increasing filters 32->64->128->256) trained from scratch on
   FER2013 with data augmentation (rotation, shift, zoom, flip).
3. **Real-Time Pipeline** - Each detected face is cropped, resized to 48x48,
   normalized, and classified; the predicted emotion and confidence are
   overlaid on the live video feed alongside a bounding box.

## Results

| Metric | Value |
|---|---|
| Overall Test Accuracy | 61.9% |
| Best Class (Happy) | F1-score 0.85 |
| Weakest Class (Fear) | F1-score 0.41 |

FER2013 is a known-difficult dataset (state-of-the-art models reach ~70-75%)
due to low resolution and inherently ambiguous expressions. Per-class
analysis (see `confusion_matrix.png`) revealed the model's main confusion is
between Happy and Surprise - both classes share visual features like wide
eyes and open mouths, a limitation also documented in FER2013 research
literature.

## How to Run

```bash
pip install -r requirements.txt
python train_emotion_cnn.py           # trains and saves the CNN
python Webcam_emotion_detection.py     # live webcam demo
python confusion_matrix.py             # per-class error analysis
```

Press `q` to quit the webcam window.

## Tech Stack

- Python, TensorFlow/Keras
- OpenCV (Haar Cascade face detection, webcam capture, visualization)
- Scikit-learn (confusion matrix, classification report)