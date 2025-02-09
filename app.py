import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained hand gesture model
MODEL_PATH = "hand_gesture_model.h5"  # Update with your trained model path
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

def main():
    st.title("Hand Gesture Recognition")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = predict_gesture(hand_landmarks)
                h, w, _ = frame.shape
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        FRAME_WINDOW.image(frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
