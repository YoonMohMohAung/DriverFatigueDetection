import cv2
import time

# ===== Internal modules =====
from landmarks import get_landmarks
from yawning_detector import mouth_aspect_ratio
from eye_closure import get_ear
from head_nodding import HeadNodDetector
from fatigue_analyzer import fatigue_level
from utils import Perclos

# ==============================
# Configuration
# ==============================
VIDEO_PATH = "data/raw_videos/subject01.mp4"

EAR_THRESHOLD = 0.25          # eye closed threshold
MAR_THRESHOLD = 0.6           # yawning threshold
YAWN_TIME_THRESHOLD = 1.5     # seconds

# ==============================
# Initialize components
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

perclos_calc = Perclos()
nod_detector = HeadNodDetector()

yawn_count = 0
mar_start_time = None

start_time = time.time()

# ==============================
# Main processing loop
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    landmarks = get_landmarks(frame)

    if landmarks:
        # ------------------------------
        # 1. Eye Closure (EAR)
        # ------------------------------
        ear = get_ear(landmarks, w, h)
        perclos_calc.update(ear, EAR_THRESHOLD)

        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "Eyes Closed", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ------------------------------
        # 2. Yawning Detection (MAR)
        # ------------------------------
        mar = mouth_aspect_ratio(landmarks, w, h)

        if mar > MAR_THRESHOLD:
            if mar_start_time is None:
                mar_start_time = time.time()
            elif time.time() - mar_start_time >= YAWN_TIME_THRESHOLD:
                yawn_count += 1
                mar_start_time = None
        else:
            mar_start_time = None

        # ------------------------------
        # 3. Head Nodding Detection
        # ------------------------------
        nod_detector.update(landmarks, h)

        # ------------------------------
        # Display metrics
        # ------------------------------
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Yawns: {yawn_count}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Nods: {nod_detector.nod_count}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Driver Fatigue Monitoring", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# Final Fatigue Analysis
# ==============================
cap.release()
cv2.destroyAllWindows()

perclos_value = perclos_calc.value()
fatigue = fatigue_level(
    yawn_count=yawn_count,
    perclos=perclos_value
)

print("========== FINAL RESULT ==========")
print(f"Yawning Count : {yawn_count}")
print(f"PERCLOS       : {perclos_value:.2f}")
print(f"Head Nods     : {nod_detector.nod_count}")
print(f"Fatigue Level : {fatigue}")
print("==================================")
