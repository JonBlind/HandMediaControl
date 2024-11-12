# OpenCV Gesture-Based Media Controller ✋

A Python application that enables hands-free media control using gestures recognized through a webcam feed. Leveraging OpenCV for video capture and TensorFlow for gesture recognition, this project allows users to play, pause, change tracks, and adjust volume through simple hand gestures.

## Features

- **Real-Time Gesture Recognition**: Analyzes webcam feed in real-time, identifying hand gestures for media control.
- **Gesture-Based Controls**: Play/pause, volume adjustment, and track navigation based on specific hand movements.
- **Custom LSTM Model**: Trained using TensorFlow with custom-recorded gesture data, ensuring accurate detection.
- **MediaPipe Integration**: Tracks hand positions and gestures reliably with the MediaPipe library.

## Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/OpenCV-Gesture-Media-Controller.git
    cd OpenCV-Gesture-Media-Controller
    ```

2. **Install Dependencies**:

3. **Run the Application**:
    ```bash
    python main.py
    ```
    Follow on-screen prompts to select display mode and camera feed visibility.

## Usage 

1. **Choose Mode**: Choose between "static" and "mobile" modes for gesture-based media control.
2. **Select Camera Display**: Choose whether to show the live camera feed.
3. **Use Gestures**:
   - Open palm (Show your palm to the Camera): Play/Pause
   - Point up/down: Increase/Decrease volume
   - Swipe left/right (mobile mode, [WARNING: UNRELIABLE]) or point left/right (static mode): Previous/Next track

## How It Works 

1. **Video Capture**: OpenCV captures frames from the user’s webcam.
2. **Gesture Detection**: MediaPipe identifies hand landmarks, which are processed into feature vectors.
3. **LSTM Model**: A TensorFlow LSTM model classifies each frame sequence into gestures.
4. **Media Control**: Detected gestures trigger corresponding media actions via Windows API calls.

## Model Training 

The gesture recognition model is an LSTM neural network trained with custom-recorded hand gesture data. The training process includes:

- Data preprocessing and gesture labeling
- Sequential feature extraction
- Training the model using TensorFlow, validated with a custom evaluation set

## Dependencies 

- OpenCV
- TensorFlow
- MediaPipe
- PyWin32 (for Windows media control API)

## License

This project is licensed under the MIT License.
