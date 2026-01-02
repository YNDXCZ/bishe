import cv2
import os
import numpy as np

class DataPreprocessor:
    def __init__(self, input_dir="data/raw", output_dir="data/processed", target_size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.categories = ['good', 'bad']
        
        for cat in self.categories:
            os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    def process(self):
        print("Starting Data Preprocessing...")
        for cat in self.categories:
            path = os.path.join(self.input_dir, cat)
            if not os.path.exists(path):
                print(f"Directory not found: {path}, skipping.")
                continue
                
            files = os.listdir(path)
            print(f"Processing {len(files)} images in '{cat}'...")
            
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Resize
                img_resized = cv2.resize(img, self.target_size)
                
                # Save original processed
                base_name = os.path.splitext(file)[0]
                self.save_image(img_resized, cat, f"{base_name}_resized")
                
                # Augment: Flip Horizontal (maybe not good for asymmetric posture? lets skip flip for now if direction matters, but usually for "bad" posture general detection it might be fine. Safe to skip if unsure.)
                # Actually, "leaning" might be direction specific. Let's do rotation instead.
                
                # Augment: Rotation +/- 10 degrees
                self.augment_rotation(img_resized, cat, base_name, 10)
                self.augment_rotation(img_resized, cat, base_name, -10)
                
                # Augment: Brightness
                self.augment_brightness(img_resized, cat, base_name, 30)

    def augment_rotation(self, img, category, base_name, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        self.save_image(rotated, category, f"{base_name}_rot{angle}")

    def augment_brightness(self, img, category, base_name, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value) # OpenCV handles saturation automatically
        final_hsv = cv2.merge((h, s, v))
        bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        self.save_image(bright_img, category, f"{base_name}_bright")

    def save_image(self, img, category, name):
        filename = f"{self.output_dir}/{category}/{name}.jpg"
        cv2.imwrite(filename, img)

if __name__ == "__main__":
    # Ensure raw directory exists or creates dummy
    if not os.path.exists("data/raw"):
        print("No input data found at data/raw. Please run collector.py first.")
    else:
        preprocessor = DataPreprocessor()
        preprocessor.process()
