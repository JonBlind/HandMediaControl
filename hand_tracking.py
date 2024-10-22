import mediapipe as mp
import numpy as np
import cv2
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Code to capture hand position and tracking is given by
# MediaPipe in their original documentation
# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
def video_capture():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("HandTracking")
    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # If capture doesn't succeed, OpenCV failed to acccess the Camera, exit and print error.
        if not cap.isOpened():
            print("Error Occurred in Opening Camera")
            exit()
    
    # Start time of the video recording
        start_time = time.time()

        while True:
            ret, frame = cap.read()

        # If ret returns False, that indicates an error with retrieving a frame from cap.read.
            if not ret:
                print("Failure to retrieve frame. Ignoring it.")
                break

            # Documentation claims marking frames as unwritable improves performance.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)


            # Now that we processed the frame, we can make it writable again.
            frame.flags.writeable = True

            # Recreate frame and use it to draw each landmark.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            


        

        # Show the captured frame
            cv2.imshow("HandTracking", cv2.flip(frame, 1))

        # Exit if ESC is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII for the Esc key
                break 
    
    cap.release()
    cv2.destroyAllWindows()


video_capture()



