#################################################################################
#
#  Project Name  : Real-Time Emotion Detection (CNN + OpenCV)
#  Description   : Opens the live webcam feed, uses OpenCV's Haar Cascade
#                   classifier to detect faces in each frame, crops each
#                   detected face, preprocesses it to match the trained CNN's
#                   expected input (48x48 grayscale, normalized), predicts
#                   the emotion, and overlays a bounding box with the
#                   predicted emotion and confidence on the live video feed.
#  Date          : 24-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "emotion_cnn_model.keras"
IMG_SIZE = 48

EMOTION_CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


#################################################################################
#
# Function Name : load_trained_model
# Input :         model_path
# Description :   Loads the emotion classification CNN trained by
#                 train_emotion_cnn.py
# Return Value :  Loaded Keras model
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from : ", model_path)
    return model


#################################################################################
#
# Function Name : load_face_detector
# Description :   Loads OpenCV's pretrained Haar Cascade classifier for
#                 frontal face detection - this is a classical (non-deep-
#                 learning) computer vision technique that ships built into
#                 OpenCV, requiring no separate download or training
# Return Value :  cv2.CascadeClassifier object
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("Haar Cascade face detector loaded.")
    return face_cascade


#################################################################################
#
# Function Name : preprocess_face
# Input :         face_roi
# Description :   Converts a cropped face region (already grayscale, from
#                 the full-color frame) into the model's expected input
#                 format: 48x48 size, normalized 0-1, with batch and
#                 channel dimensions added
# Return Value :  Preprocessed face ready for model.predict()
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def preprocess_face(face_roi):
    resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return reshaped


#################################################################################
#
# Function Name : predict_emotion
# Input :         model, processed_face
# Description :   Runs prediction on a preprocessed face and returns the
#                 predicted emotion label with its confidence score
# Return Value :  predicted_emotion (string), confidence (float)
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def predict_emotion(model, processed_face):
    predictions = model.predict(processed_face, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    predicted_emotion = EMOTION_CLASSES[predicted_index]

    return predicted_emotion, confidence


#################################################################################
#
# Function Name : run_webcam_loop
# Input :         model, face_cascade
# Description :   Opens the default webcam, detects faces in each frame
#                 using Haar Cascade, classifies the emotion of each
#                 detected face, and draws a bounding box with the predicted
#                 emotion label on the live video feed. Press 'q' to quit.
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def run_webcam_loop(model, face_cascade):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]

            processed = preprocess_face(face_roi)
            predicted_emotion, confidence = predict_emotion(model, processed)

            label_text = f"{predicted_emotion}: {confidence*100:.1f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        cv2.imshow("Real-Time Emotion Detection (press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#################################################################################
#
# Function Name : main
# Description :   Loads the trained emotion model and Haar Cascade face
#                 detector, then starts the live webcam detection loop
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    model = load_trained_model(MODEL_PATH)
    face_cascade = load_face_detector()
    run_webcam_loop(model, face_cascade)


if __name__ == "__main__":
    main()