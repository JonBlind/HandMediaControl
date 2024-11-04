import mediapipe as mp
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from data.preprocess import PreprocessGestureData

class GestureRecognizer:
    '''
    Class responsible for actually recording the user and then processing each video frame.\n
    Will identify a gesture and print it alongside the confidence level.
    '''
    def __init__(self, model_path='data/model/gesture_model.keras', label_map_path='data/model/label_map.json', sequence_length=30):

        # Load model and label map
        self.model = keras.models.load_model(model_path)
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.reverse_label_map = {index: label for label, index in self.label_map.items()}
        self.sequence_length = sequence_length
        self.sequence = []

        # Initialize MediaPipe hand tracking
        self.mp_drawing = mp.solutions.drawing_utils
        self.preprocessor = PreprocessGestureData(None, None, sequence_length=self.sequence_length)
        self.hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame_for_gesture(self, frame):
        '''
        <p> Method to process a frame, update the sequence, and then return the gesture identified.</p>
        
        <pre>process_frame_for_gesture(self, frame)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>frame</code>, <code>frame</code>Extracted frame outputted by OpenCV.</li>
        </ul>

        <strong>Return:</strong>\n
        <code>gesture, confidence</code>\n
        <code>gesture</code>: The gesture that the program identified the user to be expressing.\n
        <code>confidence</code>: The confidence number that the model outputs corresponding to the gesture.
        '''
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # Extract landmarks and preprocess
            absolute_landmarks = [coord for landmark in hand_landmarks.landmark 
                                  for coord in (landmark.x, landmark.y, landmark.z)]
            processed_landmarks = self.preprocessor.construct_landmark_vector(
                absolute_landmarks, handedness, self.sequence[-1][:3] if self.sequence else None
            )

            self.sequence.append(processed_landmarks)

            # Ensure sequence is 30 frames
            if len(self.sequence) > self.sequence_length:
                self.sequence.pop(0)

            # Predict if sequence is ready
            if len(self.sequence) == self.sequence_length:
                prediction = self.model.predict(np.expand_dims(self.sequence, axis=0))
                gesture_idx = np.argmax(prediction)
                gesture = self.reverse_label_map[gesture_idx]
                confidence = np.max(prediction)
                return gesture, confidence

        return None, None

    def capture_video_feed(self, display):
        '''
        <p> Method to capture video feed via openCV.</p>
        
        <pre>capture_video_feed(self, display)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>display</code>, <code>boolean</code>Should the video feed be played back to the user?</li>
        </ul>

        <strong>Return:</strong>\n
        No Return, it is a loop that displays video feed.
        '''
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Error: Could not open camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not retrieve frame.")
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame
            gesture, confidence = self.process_frame_for_gesture(frame)

            if gesture and confidence >= 0.75:  
                yield gesture, confidence

                if display:
                    cv2.putText(frame, f'{gesture} ({confidence:.2f})', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)

            # Display the frame if requested
            if display:
                cv2.imshow("HandTracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    break

        cap.release()
        cv2.destroyAllWindows()

