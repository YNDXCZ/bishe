import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def find_position(self, img):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # cx, cy are pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                # visibility is also useful
                lm_list.append([id, cx, cy, lm.z, lm.visibility])
        return lm_list

    def calculate_angle(self, p1, p2, p3):
        # p1, p2, p3 are [x, y] coordinates
        # Calculate angle at p2
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / denom
        # Clip to handle floating point errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def calculate_shoulder_slope(self, lm_list):
        # lm_list: [id, x, y, z, visibility]
        # Left(11), Right(12)
        
        # Get Y coordinates
        y11 = lm_list[11][2]
        y12 = lm_list[12][2]
        
        # Simple slope: difference in height
        # Positive: Right shoulder lower (Leaning Right)
        # Negative: Left shoulder lower (Leaning Left)
        return y12 - y11 

    def calculate_head_deviation(self, lm_list):
        # Nose(0), Shoulders(11, 12)
        nose_x = lm_list[0][1]
        x11 = lm_list[11][1]
        x12 = lm_list[12][1]
        
        # Midpoint of shoulders
        shoulder_center_x = (x11 + x12) / 2
        
        # Deviation: Nose X - Center X
        # Standardize by shoulder width to make it scale-invariant
        shoulder_width = abs(x11 - x12)
        if shoulder_width == 0: return 0
        
        deviation_ratio = (nose_x - shoulder_center_x) / shoulder_width
        return deviation_ratio

    def get_landmark_z(self, lm_list, id):
        # MediaPipe Z is relative to hips. 
        # Negative Z: Closer to camera
        # Positive Z: Further from camera
        return lm_list[id][3]

