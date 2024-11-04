

def main():
    # Get user input first to avoid unintended camera activation or model loading.
    mode = input("Enter Number To Choose Mode:\n 1.'static' OR 2.'mobile'\n").strip()

    if mode not in ["1", "2"]:
        print("Invalid Input. Defaulting to 'static'.")
        mode = "static"
    else:
        mode = "mobile" if mode == "2" else "static"
    
    display_camera = input("Display Live Camera Feed? (y/n)\n").strip().lower()

    displaySetting = True if display_camera == 'y' else False

    # Import gesture and media classes only after user input to avoid early camera/model activation.
    from gesture_tracking import GestureRecognizer
    from media_controller import MediaController

    # Initialize recognizer and controller after all user input
    recognizer = GestureRecognizer()
    controller = MediaController(cooldown_period=2.0)

    # Start capturing and controlling media based on gestures
    for gesture, confidence in recognizer.capture_video_feed(display=displaySetting):
        if confidence >= 0.99:
            controller.control_media_on_gesture(gesture, mode)

if __name__ == "__main__":
    main()
