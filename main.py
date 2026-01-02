import sys
import traceback

def main():
    try:
        print("Initializing Application...")
        
        # CRITICAL: Import MediaPipe and OpenCV BEFORE PyQt5 to avoid DLL conflicts
        import mediapipe as mp
        print("MediaPipe imported successfully (Pre-load).")
        import cv2
        print("OpenCV imported successfully (Pre-load).")
        
        from PyQt5.QtWidgets import QApplication
        print("PyQt5 imported.")
        from gui.main_window import MainWindow
        print("MainWindow imported.")
        
        app = QApplication(sys.argv)
        print("QApplication created.")
        
        window = MainWindow()
        print("MainWindow instantiated.")
        
        window.show()
        print("Window shown. Entering event loop.")
        
        sys.exit(app.exec_())
    except Exception as e:
        print("CRITICAL ERROR DURING STARTUP:")
        traceback.print_exc()
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        print("\nError info saved to error_log.txt")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
