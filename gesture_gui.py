from PyQt6.QtWidgets import QMainWindow, QPushButton, QRadioButton, QCheckBox, QVBoxLayout, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPalette
import time
from gesture_tracking import GestureRecognizer
from media_controller import control_media_on_gesture


class GestureWorker(QThread):
    gesture_detected = pyqtSignal(str, float)

    def __init__(self, recognizer, show_display):
        super().__init__()
        self.recognizer = recognizer
        self.show_display = show_display
        self.running = True
        self.cooldown_period = 1.5
        self.last_recognized_time = 0

    def run(self):
        for gesture, confidence in self.recognizer.capture_video_feed(display=self.show_display):
            if not self.running:
                break
            current_time = time.time()
            if confidence >= 0.99 and (current_time - self.last_recognized_time >= self.cooldown_period):
                self.gesture_detected.emit(gesture, confidence)
                self.last_recognized_time = current_time

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class GestureControlApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window Initialization
        self.setWindowTitle("Gesture Media Controller")
        self.setGeometry(300, 300, 400, 300)

        # Gesture recognizer and worker thread
        self.gesture_recognizer = GestureRecognizer()
        self.worker = None
        self.tracking = False

        # Power Button
        self.power_button = QPushButton("Off", self)
        self.power_button.setCheckable(True)
        self.power_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.power_button.setFixedSize(QSize(100, 100))
        self.power_button.clicked.connect(self.toggle_tracking)
        self.update_power_button_color()

        # Static Button
        self.static_radio = QRadioButton("Static", self)
        self.static_radio.setChecked(True)

        # Mobile Button
        self.mobile_radio = QRadioButton("Mobile", self)

        # Display CheckBox
        self.display_checkbox = QCheckBox("Show Camera Feed", self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.power_button)
        layout.addWidget(self.static_radio)
        layout.addWidget(self.mobile_radio)
        layout.addWidget(self.display_checkbox)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_power_button_color(self):
        # Update the power button text and color based on tracking state
        palette = self.power_button.palette()
        color = QColor("green") if self.tracking else QColor("gray")
        palette.setColor(QPalette.ColorRole.Button, color)
        self.power_button.setPalette(palette)
        self.power_button.update()
        self.power_button.setText("On" if self.tracking else "Off")

    def toggle_tracking(self):
        if self.worker and self.worker.isRunning():
            # Stop tracking
            self.worker.stop()
            self.worker = None
            self.tracking = False
        else:
            # Start tracking
            self.start_tracking()
            self.tracking = True

        # Update button style based on tracking state
        self.update_power_button_color()

    def start_tracking(self):
        mode = "mobile" if self.mobile_radio.isChecked() else "static"
        show_display = self.display_checkbox.isChecked()

        # Initialize worker thread
        self.worker = GestureWorker(self.gesture_recognizer, show_display)
        self.worker.gesture_detected.connect(lambda gesture, confidence: control_media_on_gesture(gesture, mode))
        self.worker.start()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()
