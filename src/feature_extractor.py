import cv2
import csv
import time
import os

from landmarks import get_landmarks
from eye_closure import get_ear
from yawning_detector import mouth_aspect_ratio
from head_nodding import HeadNodDetector

VIDEO_PATH = "data/raw_videos/subject01.mp4"
OUTPUT_CSV = "data/features/features.csv"

os.makedirs("data/features", exist_ok=True)

def extract_features(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    nod_detector = HeadNodDetector()

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "ear", "mar", "nod"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            landmarks = get_landmarks(frame)

            if landmarks:
                ear = get_ear(landmarks, w, h)
                mar = mouth_aspect_ratio(landmarks, w, h)
                nod_detector.update(landmarks, h)

                writer.writerow([
                    round(time.time(), 3),
                    round(ear, 3),
                    round(mar, 3),
                    nod_detector.nod_count
                ])

    cap.release()
    print("Feature extraction complete.")

if __name__ == "__main__":
    extract_features(VIDEO_PATH, OUTPUT_CSV)
