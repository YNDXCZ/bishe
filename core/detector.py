
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

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
