import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained hand gesture model
MODEL_PATH = "hand_gesture_model.h5"  # Ensure this file is in the same directory
model = load_model(MODEL_PATH)

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define gesture labels
gesture_labels = [
    "Palm", "Fist", "Thumbs Up", "Thumbs Down", "Peace", "Rock", "Okay", "Spiderman",
    "Call Me", "Stop", "Live Long & Prosper", "Fingers Crossed", "Heart", "Pointing", "Unknown"
]

def predict_gesture(hand_landmarks):
    """Extract features from hand landmarks and predict gesture."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks).reshape(1, -1)
    prediction = model.predict(landmarks)
    return gesture_labels[np.argmax(prediction)]

def process_image(image):
    """Process an uploaded image and return the gesture prediction."""
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = predict_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def main():
    st.title("Hand Gesture Recognition")
    st.markdown("Use the camera to capture an image or upload one.")

    # Capture image from camera
    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
