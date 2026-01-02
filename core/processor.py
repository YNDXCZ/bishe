import cv2
import time
import pickle
import numpy as np
import threading
import os
import winsound
from collections import deque, Counter
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
        
        # Time-Series Analysis (Temporal Smoothing)
        self.pose_history = deque(maxlen=15)
        self.smoothed_label = "Unknown"
        
    def process_frame(self, frame):
        # 1. Detect
        frame = self.detector.find_pose(frame)
        lm_list = self.detector.find_position(frame)
        
        instant_label = "Unknown"
        confidence = 0.0

        if len(lm_list) != 0:
            # 2. Base Features & Expert Metrics
            features = self.extract_features(lm_list)
            
            # --- Expert System Metrics ---
            slope = self.detector.calculate_shoulder_slope(lm_list)
            deviation = self.detector.calculate_head_deviation(lm_list)
            
            # 3D Depth (Forward Head)
            ear_z = (self.detector.get_landmark_z(lm_list, 7) + self.detector.get_landmark_z(lm_list, 8)) / 2
            shoulder_z = (self.detector.get_landmark_z(lm_list, 11) + self.detector.get_landmark_z(lm_list, 12)) / 2
            z_diff = ear_z - shoulder_z
            
            # Rotation
            l_11_z, r_12_z = self.detector.get_landmark_z(lm_list, 11), self.detector.get_landmark_z(lm_list, 12)
            body_rotation = abs(l_11_z - r_12_z)
            is_frontal = body_rotation < 0.20

            # --- HYBRID LOGIC: Model First, Expert Second ---
            model_prediction = 1 # Default to Good
            
            if self.model_loaded:
                # 1. Verify with Trained Model (The Authority)
                feat_vector = np.array([[ 
                    features['left_neck_incline'], features['right_neck_incline'],
                    features['left_torso_incline'], features['right_torso_incline']
                ]])
                model_prediction = self.model.predict(feat_vector)[0]
                
                # Calculate Confidence
                if hasattr(self.model, "predict_proba"):
                    confidence = max(self.model.predict_proba(feat_vector)[0])
                elif hasattr(self.model, "decision_function"):
                    dist = abs(self.model.decision_function(feat_vector)[0])
                    confidence = 1 / (1 + np.exp(-dist))
            
            # --- Final Decision & Labeling ---
            
            if model_prediction == 0: # Model says BAD
                # Use Geometry to diagnose WHY it is bad
                if is_frontal and abs(slope) > 30:
                    instant_label = "Leaning Right" if slope > 0 else "Leaning Left"
                elif is_frontal and abs(deviation) > 0.20:
                    instant_label = "Head Not Centered"
                elif is_frontal and z_diff < -0.15:
                    instant_label = "Forward Head"
                else:
                    instant_label = "Slouching" # Generic Bad from Model
            
            else: # Model says GOOD
                # SAFETY NET: Did the Model miss a 3D Forward Head issue?
                # (SVM only sees 2D angles, so it often misses pure Z-axis movement)
                if is_frontal and z_diff < -0.20: # Use safe hard threshold for "Extreme" forward head
                    instant_label = "Forward Head (3D)"
                    model_prediction = 0 # Force Bad
                else:
                    instant_label = "Good"
                    if not is_frontal: instant_label = "Good (Side)"

            # --- Temporal Smoothing (Time-Series) ---
            self.pose_history.append(instant_label)
            if len(self.pose_history) > 5:
                most_common = Counter(self.pose_history).most_common(1)[0]
                self.smoothed_label = most_common[0]
                confidence = most_common[1] / len(self.pose_history)
            else:
                self.smoothed_label = instant_label
                confidence = 1.0

            # --- Alert Logic ---
            if "Good" not in self.smoothed_label and self.smoothed_label != "Unknown":
                if not self.is_bad_posture:
                    self.is_bad_posture = True
                    self.bad_posture_start_time = time.time()
                
                if self.bad_posture_start_time and (time.time() - self.bad_posture_start_time > self.alert_threshold):
                    self.trigger_alert()
            else:
                self.is_bad_posture = False
                self.bad_posture_start_time = None

            # Visuals
            color = (0, 0, 255) if "Good" not in self.smoothed_label else (0, 255, 0)
            self.draw_debug_overlay(frame, lm_list, color)
            
            # Debug Stats
            cv2.putText(frame, f"Z-Diff: {z_diff:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"Rot: {body_rotation:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if not self.model_loaded:
                cv2.putText(frame, "NO MODEL - USING HEURISTICS", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return frame, self.smoothed_label, confidence

    def draw_debug_overlay(self, img, lm_list, color):
        try:
            # 1. Shoulder Line
            x11, y11 = lm_list[11][1], lm_list[11][2]
            x12, y12 = lm_list[12][1], lm_list[12][2]
            cv2.line(img, (x11, y11), (x12, y12), color, 2)
            
            # 2. Shoulder Center
            center_x, center_y = (x11 + x12) // 2, (y11 + y12) // 2
            
            # 3. Nose Connection
            nose_x, nose_y = lm_list[0][1], lm_list[0][2]
            cv2.line(img, (center_x, center_y), (nose_x, nose_y), (255, 255, 0), 2)
            cv2.circle(img, (nose_x, nose_y), 5, (255, 255, 0), cv2.FILLED)
            
            # 4. Vertical Reference
            cv2.line(img, (center_x, center_y), (center_x, center_y - 100), (200, 200, 200), 1, cv2.LINE_AA)
        except:
            pass

    def extract_features(self, lm_list):
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
        threading.Thread(target=self.play_sound).start()
        if self.db:
            # Check if text is too long for DB column, though 'bad' is short.
            # Using the specific label might be nice, but schema says varchar(20)
            # 'Forward Head' is 12 chars. Safe.
            label_to_log = self.smoothed_label if self.smoothed_label != "Unknown" else "bad"
            # Ensure we don't exceed schema limits just in case
            if len(label_to_log) > 20: label_to_log = "bad"
                
            self.db.log_posture(self.user_id, label_to_log, 30)
            self.bad_posture_start_time = time.time() 

    def play_sound(self):
        try:
             winsound.Beep(1000, 500) 
        except:
             pass 
