import mediapipe as mp
import numpy as np
import cv2
import json
import tensorflow as tf
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


def is_swipe_motion(sequence, direction):
    x_positions = [frame[0] for frame in sequence]  # Extract x-coordinates of wrist
    
    if direction == 'right':
        for i in range(1, len(x_positions)):
            if x_positions[i] >= x_positions[i - 1]:
                return False
        return True
    
    elif direction == 'left':
        for i in range(1, len(x_positions)):
            if x_positions[i] <= x_positions[i - 1]:
                return False
        return True
    
    return False

def handle_swipe_motion(frame, gesture, confidence, sequence):
    direction = "left" if gesture == "swipe_left" else "right"

    if is_swipe_motion(sequence, direction):
        print(f"Swipe motion detected in the {direction} direction with confidence {confidence}")
        cv2.putText(frame, f'{gesture} ({confidence:.2f})', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    
    else:
        print("Swipe motion not consistent with detected direction.")
        cv2.putText(frame, f'idle ({confidence:.2f})', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)



def process_frame(frame, hands, previous_wrist_pos):
    """
    Process a single frame to extract landmarks, handedness, and construct the feature vector.

    Args:
        frame: The current video frame from which to detect the hand.
        previous_wrist_pos: The previous wrist position to calculate displacement.

    Returns:
        processed_landmarks (list): The constructed feature vector for the frame.
        updated_wrist_pos (list): The updated wrist position to pass to the next frame.
        handedness_label (str): The handedness of the detected hand ("Right" or "Left").
        landmarks_drawn (bool): Flag indicating if landmarks were successfully detected and drawn.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        # Extract absolute landmarks (21 frames * 3 coords = 63 values)
        absolute_landmarks = [coord for landmark in hand_landmarks.landmark 
                              for coord in (landmark.x, landmark.y, landmark.z)]


        # Use the helper method in PreprocessGestureData to construct the feature vector
        processed_landmarks = preprocessor.construct_landmark_vector(
            absolute_landmarks, handedness, previous_wrist_pos
        )

        # Update wrist position for the next frame
        updated_wrist_pos = absolute_landmarks[:3]

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        return processed_landmarks, updated_wrist_pos, handedness, True
    else:
        return None, previous_wrist_pos, None, False
    

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

        sequence = []  # Store a sequence of landmarks
        previous_wrist_pos = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not retrieve frame.")
                break

            # Flip frame to mirror and look like a selfie.
            frame = cv2.flip(frame, 1)

            # Process frame and extract feature vector
            processed_landmarks, updated_wrist_pos, handedness_label, landmarks_drawn = process_frame(frame, hands, previous_wrist_pos)

            # Update the wrist position
            previous_wrist_pos = updated_wrist_pos

            if landmarks_drawn and processed_landmarks:
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
                    if confidence >= 0.5 and gesture != "idle": 
                        cv2.putText(frame, f'{gesture} ({confidence:.2f})', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

            # Display the frame
            cv2.imshow("HandTracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()

video_capture()