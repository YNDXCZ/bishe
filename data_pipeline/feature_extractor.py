import cv2
import os
import pandas as pd
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.detector import PoseDetector

class FeatureExtractor:
    def __init__(self, input_dir="data/processed", output_file="data/features.csv"):
        self.input_dir = input_dir
        self.output_file = output_file
        self.detector = PoseDetector(static_image_mode=True)
        self.categories = {'good': 1, 'bad': 0}

    def process(self):
        data = []
        print("Starting Feature Extraction...")
        
        for cat, label in self.categories.items():
            path = os.path.join(self.input_dir, cat)
            if not os.path.exists(path):
                print(f"Directory not found: {path}")
                continue
                
            files = os.listdir(path)
            print(f"Extracting features from {cat}...")
            
            for file in files:
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Detect Pose
                self.detector.find_pose(img, draw=False)
                lm_list = self.detector.find_position(img)
                
                if len(lm_list) != 0:
                    features = self.extract_angles(lm_list)
                    features['label'] = label
                    features['filename'] = file
                    data.append(features)
                    
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        print(f"Features saved to {self.output_file}. Total samples: {len(df)}")

    def extract_angles(self, lm_list):
        # MediaPipe Keypoints (0-32). relevant for sitting:
        # 11: left_shoulder, 12: right_shoulder
        # 23: left_hip,      24: right_hip
        # 7: left_ear,       8: right_ear
        
        # Helper to get coords
        def get_coords(idx):
             # lm_list[idx] = [id, x, y, z, visibility]
            return [lm_list[idx][1], lm_list[idx][2]]

        # Define points
        l_shoulder = get_coords(11)
        r_shoulder = get_coords(12)
        l_ear = get_coords(7)
        r_ear = get_coords(8)
        l_hip = get_coords(23)
        r_hip = get_coords(24)
        
        # Calculate key angles
        # 1. Neck Angle (Ear-Shoulder vs Vertical) - rough approx
        # Vertical reference point (shoulder x, shoulder y - 100)
        
        # Let's compute simpler geometric features:
        # Angle: Ear - Shoulder - Hip
        left_neck_incline = self.detector.calculate_angle(l_ear, l_shoulder, l_hip)
        right_neck_incline = self.detector.calculate_angle(r_ear, r_shoulder, r_hip)
        
        # Angle: Shoulder - Hip - Vertical (Torso inclination)
        # Vertical point from hip: (hip.x, hip.y - 100)
        l_hip_vert = [l_hip[0], l_hip[1] - 100]
        r_hip_vert = [r_hip[0], r_hip[1] - 100]
        
        left_torso_incline = self.detector.calculate_angle(l_shoulder, l_hip, l_hip_vert)
        right_torso_incline = self.detector.calculate_angle(r_shoulder, r_hip, r_hip_vert)
        
        return {
            'left_neck_incline': left_neck_incline,
            'right_neck_incline': right_neck_incline,
            'left_torso_incline': left_torso_incline,
            'right_torso_incline': right_torso_incline,
            # Add raw y-diffs relative to frame height might help too?
            # normalized dist ear to shoulder?
        }

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process()
