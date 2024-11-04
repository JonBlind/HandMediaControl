import win32api
import win32con
import time

# Constants for media keys
VK_MEDIA_PLAY_PAUSE = 0xB3  # Play/pause
VK_MEDIA_NEXT_TRACK = 0xB0  # Next track
VK_MEDIA_PREV_TRACK = 0xB1  # Previous track
VK_VOLUME_UP = 0xAF         # Volume up
VK_VOLUME_DOWN = 0xAE       # Volume down

class MediaController:
    '''
    Class responsible for controlling the actual media on the PC given gestures from the tracker.\n
    Has a cooldown period to ensure that it doesnt spam.

    '''
    def __init__(self, cooldown_period=2.0):
        self.last_recognized_time = 0
        self.cooldown_period = cooldown_period

    def press_media_key(self, vk_code):
        '''
        Method Responsible for actually interacting with the media key that is triggered.\n
        ALL MEDIA KEYS WERE FOUND IN THIS LINK:\n
        https://learn.microsoft.com/uk-ua/windows/win32/inputdev/virtual-key-codes
        '''
        win32api.keybd_event(vk_code, 0, 0, 0)
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    def play_pause(self):
        self.press_media_key(VK_MEDIA_PLAY_PAUSE)

    def next_track(self):
        self.press_media_key(VK_MEDIA_NEXT_TRACK)

    def previous_track(self):
        self.press_media_key(VK_MEDIA_PREV_TRACK)

    def volume_up(self):
        self.press_media_key(VK_VOLUME_UP)

    def volume_down(self):
        self.press_media_key(VK_VOLUME_DOWN)

    def control_media_on_gesture(self, gesture, mode):
        current_time = time.time()

        if current_time - self.last_recognized_time < self.cooldown_period:
            return  # Skip if within cooldown

        # Media control logic
        if gesture == "open_palm":
            self.play_pause()
            print("Detected Open_Palm: Pausing/Playing")
        elif gesture == "point_up":
            self.volume_up()
            print("Detected Point_Up: Increasing Volume")
        elif gesture == "point_down":
            self.volume_down()
            print("Detected Point_Down: Decreasing Volume")

        if mode == "mobile":
            if gesture == "swipe_left":
                self.previous_track()
                print("Detected Swipe_Left: Previous Track")
            elif gesture == "swipe_right":
                self.next_track()
                print("Detected Swipe_Right: Next Track")
        elif mode == "static":
            if gesture == "point_left":
                self.previous_track()
                print("Detected Point_Left: Previous Track")
            elif gesture == "point_right":
                self.next_track()
                print("Detected Point_Right: Next Track")

        # Update the last recognized time
        self.last_recognized_time = current_time