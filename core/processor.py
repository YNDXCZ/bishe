import cv2
import time
import pickle
import numpy as np
import threading
import os
import winsound
from .detector import PoseDetector
from database.db_manager import DatabaseManager

class HealthProcessor:
    def __init__(self, model_path="data/posture_model.pkl", db_manager=None, user_id=1):
        self.detector = PoseDetector()
        self.db = db_manager
        self.user_id = user_id
        
        # Load Model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        else:
            print(f"Model not found at {model_path}. Using fallback logic.")
            self.model = None
            self.model_loaded = False

        # State
        self.bad_posture_start_time = None
        self.alert_threshold = 30 # seconds
        self.is_bad_posture = False
        self.last_log_time = time.time()
        
    def process_frame(self, frame):
        # 1. Detect
        frame = self.detector.find_pose(frame)
        lm_list = self.detector.find_position(frame)
        
        posture_label = "Unknown"
        confidence = 0.0

        if len(lm_list) != 0:
            # 2. Feature Extraction
            features = self.extract_features(lm_list)
            
            # 3. Predict
            if self.model_loaded:
                # Prepare feature vector (must match training columns)
                # left_neck, right_neck_incline, etc. MUST match feature_extractor keys order!
                # My feature extractor used: left_neck_incline, right_neck_incline, etc.
                # Let's match distinct names:
                feat_vector = np.array([[ 
                    features['left_neck_incline'], features['right_neck_incline'],
                    features['left_torso_incline'], features['right_torso_incline']
                ]])
                
                prediction = self.model.predict(feat_vector)[0] # 1=Good, 0=Bad
                probs = self.model.predict_proba(feat_vector)[0]
                
                if prediction == 1:
                    posture_label = "Good"
                    self.is_bad_posture = False
                    self.bad_posture_start_time = None
                else:
                    posture_label = "Bad"
                    if not self.is_bad_posture:
                        self.is_bad_posture = True
                        self.bad_posture_start_time = time.time()
                    
                    # Check threshold
                    if self.bad_posture_start_time and (time.time() - self.bad_posture_start_time > self.alert_threshold):
                        self.trigger_alert()
                
                confidence = max(probs)
            else:
                # Fallback: Simple heuristic if no model
                if features['left_neck_incline'] < 140: # Example thresholds
                     posture_label = "Bad (Heuristic)"
                else:
                     posture_label = "Good (Heuristic)"

        return frame, posture_label, confidence

    def extract_features(self, lm_list):
        # Must match feature_extractor.py EXACTLY
        def get_coords(idx): return [lm_list[idx][1], lm_list[idx][2]]
        
        l_shoulder = get_coords(11)
        r_shoulder = get_coords(12)
        l_ear = get_coords(7)
        r_ear = get_coords(8)
        l_hip = get_coords(23)
        r_hip = get_coords(24)
        
        l_hip_vert = [l_hip[0], l_hip[1] - 100]
        r_hip_vert = [r_hip[0], r_hip[1] - 100]

        left_neck = self.detector.calculate_angle(l_ear, l_shoulder, l_hip)
        right_neck = self.detector.calculate_angle(r_ear, r_shoulder, r_hip)
        left_torso = self.detector.calculate_angle(l_shoulder, l_hip, l_hip_vert)
        right_torso = self.detector.calculate_angle(r_shoulder, r_hip, r_hip_vert)
        
        return {
            'left_neck_incline': left_neck, 
            'right_neck_incline': right_neck,
            'left_torso_incline': left_torso, 
            'right_torso_incline': right_torso
        }

    def trigger_alert(self):
        # Sound in thread to not block video
        threading.Thread(target=self.play_sound).start()
        # Log to DB
        if self.db:
            self.db.log_posture(self.user_id, 'bad', 30)
            self.bad_posture_start_time = time.time() 

    def play_sound(self):
        try:
             winsound.Beep(1000, 500) 
        except:
             pass 
