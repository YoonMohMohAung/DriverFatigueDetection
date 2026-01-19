import cv2
import csv
import time
import os

from landmarks import get_landmarks
from eye_closure import get_ear
from yawning_detector import mouth_aspect_ratio
from head_nodding import HeadNodDetector

RAW_VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print(f"[OK] Features extracted: {output_csv}")

def process_all_videos():
    for label in os.listdir(RAW_VIDEO_DIR):
        label_dir = os.path.join(RAW_VIDEO_DIR, label)

        if not os.path.isdir(label_dir):
            continue

        for video_file in os.listdir(label_dir):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(label_dir, video_file)

            output_csv = os.path.join(
                OUTPUT_DIR,
                f"{label}_{os.path.splitext(video_file)[0]}.csv"
            )

            extract_features(video_path, output_csv)

if __name__ == "__main__":
    process_all_videos()
