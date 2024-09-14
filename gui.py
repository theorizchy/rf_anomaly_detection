import sys
import re
import subprocess
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTextEdit, QPushButton, QHBoxLayout, QGridLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QBrush

# Custom widget to show a circle (red/green status indicator)
class CircleIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.color = QColor('red')  # Default to red (not running)
        self.setFixedSize(20, 20)   # Set a fixed size for the circle
        self.setStyleSheet('border-radius: 10px; background-color: grey;')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(0, 0, self.width(), self.height())  # Draw a circle with the widget's size

    def set_green(self):
        self.color = QColor('green')
        self.update()

    def set_red(self):
        self.color = QColor('red')
        self.update()


# Sample function to simulate parsing the output with error handling
def parse_output(output):
    result = {}
    lines = output.strip().split('\n')

    # Handle the "[KNOWN]" or "[UNKNOWN]" line
    for line in lines:
        if "[KNOWN]" in line or "[UNKNOWN]" in line:
            result['status'] = 'KNOWN' if "[KNOWN]" in line else 'UNKNOWN'
            result['similar_camera'] = line.split('camera_')[1].split(' ')[0]
            result['similarity_score'] = line.split('score: ')[1].strip('%')

    # Handle the "Saved data and prediction..." line
    for line in lines:
        if "Saved data" in line:
            parts = line.split(', ')
            if len(parts) >= 2:
                result['timestamp'] = parts[0].split('at ')[1]
                result['camera_status'] = parts[1].split(': ')[1].split(' ')[0]
                result['confidence'] = parts[1].split(': ')[2].split('%')[0]
            else:
                result['timestamp'] = "Unknown"
                result['camera_status'] = "Unknown"
                result['confidence'] = "Unknown"
    return result


# Thread to handle reading output in real-time
class OutputReaderThread(QThread):
    new_output = pyqtSignal(str)

    def run(self):
        command = [
            r"c:\\Users\\theor\\OneDrive\\Desktop\\Master\\Project\\rf_anomaly_detection\\venv\\Scripts\\python.exe",
            "-u", "realtime_anomaly_detector.py", "--simulated",
            r"c:/Users/theor/OneDrive/Desktop/Master/Project/rf_anomaly_detection/output_file/clearwrite_captured_SWEEP_REC_2024-07-07 20h05m06s_raspi4_a_on.csv"
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self.new_output.emit(output)


class CameraDataGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the layout and labels
        self.main_layout = QVBoxLayout()

        # Create a grid layout for the title and status circle
        title_layout = QGridLayout()

        # Add status circle at top-right corner
        self.status_circle = CircleIndicator()
        title_layout.addWidget(self.status_circle, 0, 1, alignment=Qt.AlignRight)

        # Add a title at the top
        title_label = QLabel("Camera Data Logger")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_layout.addWidget(title_label, 0, 0)

        self.main_layout.addLayout(title_layout)

        # Labels for status and similarity
        label_layout = QVBoxLayout()

        # Horizontal layout for 'Camera Status' label and value
        camera_status_layout = QHBoxLayout()
        self.camera_status_label = QLabel("Camera Status: ")
        self.camera_status_value = QLabel("")  # Dynamic value
        camera_status_layout.addWidget(self.camera_status_label)
        camera_status_layout.addWidget(self.camera_status_value)

        # Horizontal layout for 'Most Similar Camera' label, value, and similarity percentage
        similar_camera_layout = QHBoxLayout()
        self.similar_camera_label = QLabel("Most Similar Camera: ")
        self.similar_camera_value = QLabel("")  # Dynamic value
        similar_camera_layout.addWidget(self.similar_camera_label)
        similar_camera_layout.addWidget(self.similar_camera_value)

        label_layout.addLayout(camera_status_layout)
        label_layout.addLayout(similar_camera_layout)

        self.main_layout.addLayout(label_layout)

        # Add other labels with their own horizontal layouts
        self.timestamp_label = QLabel("Timestamp: ")
        self.timestamp_value = QLabel("")  # Dynamic value
        timestamp_layout = QHBoxLayout()
        timestamp_layout.addWidget(self.timestamp_label)
        timestamp_layout.addWidget(self.timestamp_value)
        self.main_layout.addLayout(timestamp_layout)

        self.camera_on_off_label = QLabel("Camera: ")
        self.camera_on_off_value = QLabel("")  # Dynamic value
        camera_status_layout = QHBoxLayout()
        camera_status_layout.addWidget(self.camera_on_off_label)
        camera_status_layout.addWidget(self.camera_on_off_value)
        self.main_layout.addLayout(camera_status_layout)

        self.confidence_label = QLabel("Confidence: ")
        self.confidence_value = QLabel("")  # Dynamic value
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_value)
        self.main_layout.addLayout(confidence_layout)

        # Add a spacer item to create vertical space
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Ignored, QSizePolicy.Ignored))

        # Add a logger window title
        logger_title_label = QLabel("Log Output")
        logger_title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.main_layout.addWidget(logger_title_label)

        # Add a QTextEdit to display the running logs
        self.log_window = QTextEdit(self)
        self.log_window.setReadOnly(True)
        self.main_layout.addWidget(self.log_window)

        # Add a button to trigger capture
        self.capture_button = QPushButton("Start Capture")
        self.capture_button.clicked.connect(self.start_capture)
        self.main_layout.addWidget(self.capture_button)

        self.setLayout(self.main_layout)

        self.output_thread = OutputReaderThread()
        self.output_thread.new_output.connect(self.handle_new_output)

        # Create a timer to handle label color resetting
        self.reset_timer = QTimer(self)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.reset_label_styles)

        # Set default font for labels and dynamic values
        label_font = QFont("Arial", 12)
        dynamic_font = QFont("Arial", 12)

        # Set bigger fonts for labels and dynamic values
        self.camera_status_label.setFont(label_font)
        self.similar_camera_label.setFont(label_font)
        self.timestamp_label.setFont(label_font)
        self.camera_on_off_label.setFont(label_font)
        self.confidence_label.setFont(label_font)

        self.camera_status_value.setFont(dynamic_font)
        self.similar_camera_value.setFont(dynamic_font)

        self.timestamp_value.setFont(dynamic_font)
        self.camera_on_off_value.setFont(dynamic_font)
        self.confidence_value.setFont(dynamic_font)

    def start_capture(self):
        self.log_window.clear()  # Clear log window at start
        self.output_thread.start()
        self.status_circle.set_green()  # Change circle to green (running)
        self.capture_button.setEnabled(False)  # Disable button after starting

    def handle_new_output(self, output):
        parsed_data = parse_output(output)
        self.update_labels(parsed_data)
        self.log_window.append(output.strip())  # Append log to QTextEdit

    def update_labels(self, parsed_data):
        # Update labels with new data
        if parsed_data.get('status'):
            self.camera_status_value.setText(parsed_data['status'])
        if parsed_data.get('similar_camera'):
            self.similar_camera_value.setText(f"camera_{parsed_data['similar_camera']} ({parsed_data['similarity_score']}%)")
        if parsed_data.get('timestamp'):
            self.timestamp_value.setText(parsed_data['timestamp'])
        if parsed_data.get('camera_status'):
            self.camera_on_off_value.setText(parsed_data['camera_status'])
        if parsed_data.get('confidence'):
            self.confidence_value.setText(f"{parsed_data['confidence']}%")

        # Change label colors to red and bold
        self.change_value_styles()

        # Start timer to reset styles after 0.5 seconds
        self.reset_timer.start(500)

    def change_value_styles(self):
        red_bold_font = QFont("Arial", 12)
        red_bold_font.setBold(True)

        self.camera_status_value.setStyleSheet('color: red')
        self.camera_status_value.setFont(red_bold_font)

        self.similar_camera_value.setStyleSheet('color: red')
        self.similar_camera_value.setFont(red_bold_font)

        self.timestamp_value.setStyleSheet('color: red')
        self.timestamp_value.setFont(red_bold_font)

        self.camera_on_off_value.setStyleSheet('color: red')
        self.camera_on_off_value.setFont(red_bold_font)

        self.confidence_value.setStyleSheet('color: red')
        self.confidence_value.setFont(red_bold_font)

    def reset_label_styles(self):
        normal_font = QFont("Arial", 12)
        self.camera_status_value.setStyleSheet('color: black')
        self.camera_status_value.setFont(normal_font)

        self.similar_camera_value.setStyleSheet('color: black')
        self.similar_camera_value.setFont(normal_font)

        self.timestamp_value.setStyleSheet('color: black')
        self.timestamp_value.setFont(normal_font)

        self.camera_on_off_value.setStyleSheet('color: black')
        self.camera_on_off_value.setFont(normal_font)

        self.confidence_value.setStyleSheet('color: black')
        self.confidence_value.setFont(normal_font)


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CameraDataGUI()
    gui.setGeometry(200, 200, 1000, 1000)  # Set the window size
    gui.show()
    sys.exit(app.exec_())
