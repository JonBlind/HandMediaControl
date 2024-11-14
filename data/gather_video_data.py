import mediapipe as mp
import cv2
import time
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

'''
swipe left = rewind
swipe right = skip
point up = volume up
point down = volume down
Open Palm = pause/resume

'''

# Mapping numbers to gesture labels
gesture_options = {
    '1': 'open_palm',
    '2': 'swipe_left',
    '3': 'swipe_right',
    '4': 'point_up',
    '5': 'point_down',
    '6': 'point_left',
    '7': 'point_right',
    '8': 'idle'
}


def get_next_index(gesture_label):
    '''
    Method to obtain the next index to be placed inside a given JSON data file.

    Goes into gesture_label's associated JSON file and finds the last index, adding 1 to it.
    '''

    filename = f"gesture_data/{gesture_label}_data.json"

    # If the path exists, we open it in read only mode
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)

        # Grab the index value for every item in existing_data and return the largest one.
        # Then add 1.
        max_index = max(item.get('index', -1) for item in existing_data)
        return max_index + 1
    
    # Return 0 if no file exists since it will be the first input after creation.
    return 0



def save_sequence_to_json(data, gesture_label):
    '''
    This function stores a sequence of frames representing a gesture into its respective JSON file.

    Arguments:
        data (Dictionary): Information to save into a JSON file (index, gesture, sequence data).

    Doesn't return anything, but directly creates or adds a data to a JSON file.
'''
    filename = f"gesture_data/{gesture_label}_data.json"
    
    # Load existing data or initialize empty array
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(data)
    
    # Write updated data back to the gesture-specific JSON file
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)
    

def video_capture_data_gather():
    '''
    Main method. Responsible for capturing the camera feed,\n
    asking the user to choose a gesture and start recording,\n
    and then asks for confirmation to save.\n

    Obviously would've been much easier if I just took photos,\n
    but this is best for the privacy of the people I ask to help.
'''
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("HandTracking")
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        capturing = False  # Flag for data capture
        sequence_data = []
        gesture_label = None
        confirmation_mode = False

        print("Instructions:\n- Press keys 1-5 to select a gesture\n- Press 'p' to start capturing\n- Press 'ESC' to exit")

        while True:
            ret, frame = cap.read()

            # If ret returns False, that indicates an error with retrieving a frame from cap.read.
            if not ret:
                print("Failure to retrieve frame. Ignoring it.")
                break 


            # Documentation claims marking frames as unwritable improves performance.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame) if capturing else None

            if capturing and results.multi_hand_landmarks:
                frame.flags.writeable = True
                frame_index = len(sequence_data)
                frame_timestamp = time.time() - start_time
            
                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    # Grab and store the hand that is being tracked.
                    handedness = results.multi_handedness[index].classification[0].label

                    # FOR SOME REASON IT ALWAYS RETURNS THE OPPOSITE HAND??
                    # So im just relabeling them.
                    # The video being mirrored doesn't change this either.
                    handedness = "Left" if handedness == "Right" else "Right"

                
                    # Create an array to collect data on landmarks. For every landmark
                    # We grab their x,y,z coordinates and store them.
                    landmarks = []

                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                    sequence_data.append({
                        "frame_index": frame_index,
                        "timestamp": frame_timestamp,
                        "handedness": handedness,
                        "landmarks": landmarks
                    })

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    

            # Convert camera to normal colors.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
            # Display instructions on the frame
            instructions = [
                "Press keys 1-5 to select a gesture:",
                "1 - Open Palm | 2 - Swipe Left | 3 - Swipe Right",
                "4 - Point Up | 5 - Point Down | 6 - Point Left",
                "7 - Point Right | 8 - Idle",
                f"Selected gesture: {gesture_label if gesture_label else 'None'}",
                "Press 'p' to start capturing, 'ESC' to quit"
            ]

            if confirmation_mode:
                instructions = [
                    "Confirm saving this capture:",
                    "Press 'y' to save or 'n' to discard."
                ]

            # Show instructions on a flipped frame
            display_frame = cv2.flip(frame, 1)

            for i, text in enumerate(instructions, start=1):
                cv2.putText(display_frame, text, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("HandTracking", display_frame)

            # Check key press in OpenCV window
            key = cv2.waitKey(1) & 0xFF
            
            # Confirmation mode logic.
            # Ask the user to confirm if they want the just recorded sequence saved
            if confirmation_mode:
                if key == ord('y'):
                    data = {
                        'index': get_next_index(gesture_label),
                        'gesture': gesture_label,
                        'sequence_data': sequence_data
                    }
                    save_sequence_to_json(data, gesture_label)
                    print(f"Data saved for gesture: {gesture_label}")
                    confirmation_mode = False
                    sequence_data = []
                elif key == ord('n'):
                    print("Data discarded.")
                    confirmation_mode = False
                    sequence_data = []
                continue

            # Select gesture based on number keys
            for i in range (1, 9):
                if key in [ord(str(i))]:
                    gesture_label = gesture_options[chr(key)]
                    print(f"Selected gesture: {gesture_label}")

            # Start capture on 'p'
            if key == ord('p') and gesture_label:
                print(f"Starting capture for gesture: {gesture_label}")
                capturing = True
                sequence_data = []
                start_time = time.time()

            # Stop capture after 2 seconds
            if capturing and (time.time() - start_time) >= 1.25:
                capturing = False
                print("Capture complete. Reviewing data...")
                confirmation_mode = True

            # Quit program on ESC key
            if key == 27:  # ESC
                print("Exiting program.")
                break

    cap.release()
    cv2.destroyAllWindows()

video_capture_data_gather()