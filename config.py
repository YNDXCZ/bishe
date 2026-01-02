import os

# Database Configuration
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = '030608' # Change this to your MySQL password
DB_NAME = 'posture_health'

# Camera Configuration
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Algorithm Configuration
ALERT_THRESHOLD_SECONDS = 30
MODEL_PATH = os.path.join("data", "posture_model.pkl")

# Paths
DATA_RAW = os.path.join("data", "raw")
DATA_PROCESSED = os.path.join("data", "processed")
