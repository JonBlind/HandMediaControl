import mediapipe as mp
import numpy as np
import cv2
import json
from tensorflow import keras
from data.preprocess import PreprocessGestureData

# Load the model and label map
model = keras.models.load_model('data/model/gesture_model.keras')

with open('data/model/label_map.json', 'r') as f:
    label_map = json.load(f)

# Invert the key and value in this mapping.
reverse_label_map = {index: label for label, index in label_map.items()}

# Initialize MediaPipe Hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
preprocessor = PreprocessGestureData(None, None, sequence_length=30)

def video_capture():
    '''
    Method responsible for actually capturing footage and running the model to determine what gestures are being done
    on live feed. Essentially a copy of the gather_video_data iteration, except here we process the landmarks and ask the
    model to output us a gesture label and confidence label.
    '''
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Setting up MediaPipe hands tracking
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        sequence = []  # To store a sequence of landmarks
        previous_wrist_pos = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not retrieve frame.")
                break
            
            # Flip frame for a mirror effect
            frame.flags.writeable = False
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            
            # If we actually see results in the landmarks
            if results.multi_hand_landmarks:
                frame.flags.writeable = True
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label

                landmarks = []

                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Mediapipe is goofy and has the hand label inverted so fix it.
                handedness_label = "Left" if handedness == "Right" else "Right"

                wrist_displacement = preprocessor.calculate_wrist_displacement(landmarks[:3], previous_wrist_pos)

                previous_wrist_pos = landmarks[:3]
                    
                # Preprocess and add to sequence
                processed_landmarks = landmarks + [1 if handedness_label == "Right" else 0] + wrist_displacement
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

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