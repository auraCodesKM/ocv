import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

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

def main():
    st.title("Live Gesture Recognition using MediaPipe")
    st.markdown(
        """
        <style>
        .big-font {
            font-size:30px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        gesture_detected = "None"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_detected = recognize_gesture(hand_landmarks)
                cv2.putText(frame, gesture_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        FRAME_WINDOW.image(frame, channels="BGR")
        st.markdown(f'<p class="big-font">Detected Gesture: {gesture_detected}</p>', unsafe_allow_html=True)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
