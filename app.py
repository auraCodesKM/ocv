import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define gesture classifier
GESTURES = {
    "Fist": lambda lm: all(lm[i][1] > lm[i + 1][1] for i in range(5, 20, 4)),
    "Palm": lambda lm: all(lm[i][1] < lm[i + 1][1] for i in range(5, 20, 4)),
    "Thumbs Up": lambda lm: lm[4][0] < lm[3][0] and all(lm[i][1] > lm[i + 1][1] for i in range(5, 17, 4)),
    "Thumbs Down": lambda lm: lm[4][0] > lm[3][0] and all(lm[i][1] < lm[i + 1][1] for i in range(5, 17, 4)),
    "Peace": lambda lm: lm[8][1] < lm[6][1] and lm[12][1] < lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1],
    "OK": lambda lm: lm[8][0] > lm[6][0] and lm[12][0] > lm[10][0] and lm[16][0] > lm[14][0] and lm[20][0] > lm[18][0],
    "Rock": lambda lm: lm[8][1] < lm[6][1] and lm[12][1] > lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] < lm[18][1],
}

def recognize_gesture(landmarks):
    lm = [(lm.x, lm.y) for lm in landmarks.landmark]
    for gesture, check in GESTURES.items():
        if check(lm):
            return gesture
    return "Unknown"

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror the image

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        gesture_detected = "None"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_detected = recognize_gesture(hand_landmarks)
                cv2.putText(img, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        return img

# Streamlit UI
st.title("Live Gesture Recognition using MediaPipe")
st.write("Enable camera access and perform gestures!")

webrtc_streamer(key="gesture-detection", video_transformer_factory=VideoTransformer)
