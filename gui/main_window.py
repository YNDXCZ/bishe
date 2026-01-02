import sys
import cv2
import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTabWidget, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Project Imports
from core.processor import HealthProcessor
from database.db_manager import DatabaseManager
import config

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_status_signal = pyqtSignal(str, str) # Label, Confidence

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = True

    update_stats_signal = pyqtSignal(str) # FPS/Latency

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_ID)
        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame.copy() # Store for capture
                # Process Frame (Detect + Predict)
                frame, label, conf = self.processor.process_frame(frame)
                
                # Emit Status
                self.update_status_signal.emit(label, f"{conf:.2f}")

                # Convert to Qt Image
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            
            # Subtracted sleep to measure pure processing latency involves more complex logic, 
            # but for "System Latency", end-to-end time is what matters.
            process_time = time.time() - start_time
            latency_ms = process_time * 1000
            fps = 1.0 / process_time if process_time > 0 else 0
            
            self.update_stats_signal.emit(f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms")
            
            # Adjust sleep to maintain cap but not double sleep
            # simple sleep for stability
            if process_time < 0.033:
                 time.sleep(0.033 - process_time)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Office Health Reminder System")
        self.resize(1000, 700)

        # Initialize Backend
        self.db = DatabaseManager() # Uses config defaults
        
        # Ensure default user exists
        current_user_id = self.db.add_user("admin")
        if not current_user_id:
            current_user_id = 1 # Fallback, though logging might fail if DB connection is broken
            
        self.processor = HealthProcessor(db_manager=self.db, user_id=current_user_id)

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Tabs
        self.tabs = QTabWidget()
        self.monitor_tab = QWidget()
        self.report_tab = QWidget()
        self.tabs.addTab(self.monitor_tab, "Monitor")
        self.tabs.addTab(self.report_tab, "Reports")
        self.layout.addWidget(self.tabs)

        self.setup_monitor_tab()
        self.setup_report_tab()

    def setup_monitor_tab(self):
        layout = QHBoxLayout(self.monitor_tab)
        
        # Left: Video
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        # Right: Controls & Status
        right_panel = QVBoxLayout()
        
        self.status_label = QLabel("Status: Unknown")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        right_panel.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: gray;")
        right_panel.addWidget(self.fps_label)

        self.btn_start = QPushButton("Start Monitoring")
        self.btn_start.clicked.connect(self.start_video)
        right_panel.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Monitoring")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        right_panel.addWidget(self.btn_stop)

        right_panel.addSpacing(20)
        
        # Data Collection Section
        self.lbl_collect = QLabel("Data Collection:")
        self.lbl_collect.setStyleSheet("font-weight: bold;")
        right_panel.addWidget(self.lbl_collect)
        
        self.btn_good = QPushButton("Capture GOOD Posture")
        self.btn_good.setStyleSheet("background-color: #d4edda; color: #155724;") 
        self.btn_good.clicked.connect(lambda: self.capture_data('good'))
        right_panel.addWidget(self.btn_good)
        
        self.btn_bad = QPushButton("Capture BAD Posture")
        self.btn_bad.setStyleSheet("background-color: #f8d7da; color: #721c24;")
        self.btn_bad.clicked.connect(lambda: self.capture_data('bad'))
        right_panel.addWidget(self.btn_bad)
        
        self.btn_retrain = QPushButton("Retrain Model")
        self.btn_retrain.clicked.connect(self.retrain_model)
        self.btn_retrain.setEnabled(False) # Enable logic later or keep manual
        right_panel.addWidget(self.btn_retrain)

        right_panel.addStretch()
        
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        right_panel.addWidget(self.btn_settings)
        
        layout.addLayout(right_panel)

    def capture_data(self, label):
        if not hasattr(self, 'thread') or not self.thread.isRunning() or not hasattr(self.thread, 'last_frame'):
            QMessageBox.warning(self, "Warning", "Please start monitoring first.")
            return

        import os
        from datetime import datetime
        
        # Determine path
        # config.DATA_RAW points to data/raw
        # We need data/raw/good or data/raw/bad
        save_dir = os.path.join("data", "raw", label)
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Save Frame
        cv2.imwrite(filepath, self.thread.last_frame)
        print(f"Captured {label} sample: {filepath}")
        
        # Flash status to visually confirm
        original_text = self.status_label.text()
        self.status_label.setText(f"Saved: {label.upper()}!")
        QTimer.singleShot(1000, lambda: self.status_label.setText(original_text))

    def retrain_model(self):
        # Placeholder for now, or we can implement calling the scripts
        QMessageBox.information(self, "Info", "Please run 'python data_pipeline/feature_extractor.py' and 'python data_pipeline/train_model.py' in terminal to update the model.")

    def open_settings(self):
        from gui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec_():
            settings = dialog.get_settings()
            # Update Processor Settings
            self.processor.alert_threshold = settings['threshold']
            # Restart video if camera changed? For now just log
            print(f"Settings Updated: {settings}")


    def setup_report_tab(self):
        layout = QVBoxLayout(self.report_tab)
        
        self.btn_refresh = QPushButton("Refresh Data")
        self.btn_refresh.clicked.connect(self.plot_charts)
        layout.addWidget(self.btn_refresh)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def start_video(self):
        self.thread = VideoThread(self.processor)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.update_stats_signal.connect(self.update_stats)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_video(self):
        if hasattr(self, 'thread'):
            self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.clear()

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_status(self, label, conf):
        color = "green" if label == "Good" else "red"
        self.status_label.setText(f"Status: {label} ({conf})")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: bold;")

    def update_stats(self, stats_text):
        self.fps_label.setText(stats_text)

    def plot_charts(self):
        self.figure.clear()
        
        # Dummy data if DB empty
        data = self.db.get_stats(user_id=1, days=7)
        if not data:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No Data Available", ha='center')
        else:
            # Process data for chart
            # date, posture_type, duration
            # We want stacked bar chart of Good vs Bad duration per day
            dates = sorted(list(set(d['date'] for d in data)))
            good_durations = []
            bad_durations = []
            
            for date in dates:
                g = next((d['total_duration'] for d in data if d['date'] == date and d['posture_type'] == 'good'), 0)
                b = next((d['total_duration'] for d in data if d['date'] == date and d['posture_type'] == 'bad'), 0)
                good_durations.append(g / 60) # mins
                bad_durations.append(b / 60)

            ax = self.figure.add_subplot(111)
            labels = [str(d) for d in dates]
            
            ax.bar(labels, good_durations, label='Good Posture')
            ax.bar(labels, bad_durations, bottom=good_durations, label='Bad Posture')
            ax.set_ylabel('Duration (Minutes)')
            ax.legend()
            
        self.canvas.draw()

    def closeEvent(self, event):
        self.stop_video()
        event.accept()
