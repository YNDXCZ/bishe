
import cv2
import os
import time

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        self.categories = ['good', 'bad']
        for cat in self.categories:
            os.makedirs(os.path.join(output_dir, cat), exist_ok=True)
        
        self.cap = cv2.VideoCapture(0)
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        print("Starting Data Collector...")
        print("Press 'g' to save GOOD posture.")
        print("Press 'b' to save BAD posture.")
        print("Press 'q' to quit.")
        
        count = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to access camera.")
                break

            cv2.putText(frame, f"Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.save_frame(frame, 'good')
                count += 1
            elif key == ord('b'):
                self.save_frame(frame, 'bad')
                count += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def save_frame(self, frame, category):
        timestamp = int(time.time() * 1000)
        filename = f"{self.output_dir}/{category}/{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {category} image: {filename}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
