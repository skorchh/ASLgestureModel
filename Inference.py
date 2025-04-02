import time
from collections import deque

import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
import pygetwindow as gw
import pyautogui

from Model import CNN  
from Preprocessing import transform

# Load the trained ASL recognition model
model = CNN()
model.load_state_dict(torch.load('asl_model.pth'))
model.eval()

# Mapping model output labels to characters (0-9 and a-z)
label_to_char = {i: str(i) for i in range(10)}
label_to_char.update({10 + i: chr(97 + i) for i in range(26)})

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Prediction smoothing using a buffer
prediction_buffer = deque(maxlen=5)

# Buffer to limit typing frequency
last_typed_time = 0
buffer_duration = 4  # Buffer between keystrokes

def is_textbox_active():
    """Check if a text input field is currently active."""
    active_window = gw.getActiveWindow()
    return bool(active_window)

def detect_thumb_gesture(hand_landmarks):
    """Detects a strict thumbs-up or thumbs-down gesture."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Determine if the thumb is extended
    thumb_extended = abs(thumb_tip.y - thumb_mcp.y) > 0.1

    # Check if fingers are curled 
    fingers_curled = thumb_tip.y < index_mcp.y if thumb_tip.y < thumb_ip.y else thumb_tip.y > index_mcp.y

    # Classify gesture as space (thumbs up) or backspace (thumbs down)
    if thumb_extended and fingers_curled:
        if thumb_tip.y < thumb_ip.y < thumb_mcp.y:
            return "space"
        elif thumb_tip.y > thumb_ip.y > thumb_mcp.y:
            return "backspace"
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h, w, _ = frame.shape
    black_background = np.zeros((h, w, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                black_background, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Detect thumb gestures for space/backspace
            gesture = detect_thumb_gesture(hand_landmarks)
            if gesture and is_textbox_active():
                current_time = time.time()
                if current_time - last_typed_time > buffer_duration:
                    if gesture == "space":
                        pyautogui.write(" ")
                    elif gesture == "backspace":
                        pyautogui.press("backspace")
                    last_typed_time = current_time
                    continue  # Skip letter prediction if gesture is detected

            # Extract bounding box around hand
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Convert coordinates to pixels
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Add padding to the bounding box
            padding = 50
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

            hand = black_background[y_min:y_max, x_min:x_max]

            # If a valid hand region is detected, process for ASL recognition
            if hand.size > 0:
                hand_pil = Image.fromarray(hand)
                transformed_hand = transform(hand_pil)

                with torch.no_grad():
                    output = model(transformed_hand.unsqueeze(0))
                    _, predicted = torch.max(output, 1)
                    predicted_index = predicted.item()
                    predicted_letter = label_to_char.get(predicted_index, '?')

                    # Add to prediction buffer for smoothing
                    prediction_buffer.append(predicted_letter)
                    most_common_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

                    current_time = time.time()
                    if is_textbox_active() and (current_time - last_typed_time > buffer_duration):
                        pyautogui.write(most_common_prediction)
                        last_typed_time = current_time

                    # Display the predicted letter on the frame
                    cv2.rectangle(frame, (x_min, y_min - 50), (x_min + 90, y_min), (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, f"{most_common_prediction}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Resize and display the frame for better visibility
    scale_factor = 2
    frame_resized = cv2.resize(frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor))
    cv2.imshow('Live ASL Detection', frame_resized)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
