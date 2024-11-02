import mediapipe as mp
import numpy as np
import cv2
import time
import json
from tensorflow import keras

# Load the model and label map

model = keras.models.load_model('data/model/gesture_model.keras')
with open('data/model/label_map.json', 'r') as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Initialize MediaPipe Hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def preprocess_landmarks(landmarks, handedness, wrist_displacement):
    # Add handedness and wrist displacement as extra features
    handedness_feature = [1 if handedness == "Right" else 0]
    return landmarks + handedness_feature + wrist_displacement

def video_capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Setting up MediaPipe hands tracking
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        sequence = []  # To store a sequence of landmarks
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not retrieve frame.")
                break
            
            # Flip frame for a mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract landmarks and preprocess
                    landmarks = [landmark.x for landmark in hand_landmarks.landmark] + \
                                [landmark.y for landmark in hand_landmarks.landmark] + \
                                [landmark.z for landmark in hand_landmarks.landmark]
                    
                    # Calculate wrist displacement (example, might adjust based on your specific displacement setup)
                    wrist_displacement = [landmarks[0] - landmarks[3], landmarks[1] - landmarks[4], landmarks[2] - landmarks[5]]
                    handedness_label = handedness.classification[0].label
                    
                    # Preprocess and add to sequence
                    processed_landmarks = preprocess_landmarks(landmarks, handedness_label, wrist_displacement)
                    sequence.append(processed_landmarks)

                    # Keep sequence length to 30 frames
                    if len(sequence) > 30:
                        sequence.pop(0)

                    # Predict gesture if sequence is ready
                    if len(sequence) == 30:
                        prediction = model.predict(np.expand_dims(sequence, axis=0))
                        gesture_idx = np.argmax(prediction)
                        gesture = reverse_label_map[gesture_idx]
                        confidence = np.max(prediction)

                        # Display gesture with confidence level
                        if confidence > 0.5:
                            cv2.putText(frame, f'{gesture} ({confidence:.2f})', (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Display the frame
            cv2.imshow("HandTracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()

video_capture()