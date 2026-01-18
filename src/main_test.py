import cv2
import numpy as np
import torch

from model import CNN_LSTM
from landmarks import get_landmarks
from utils import compute_ear, compute_mar, compute_head_nod

# ======================
# Config
# ======================
MODEL_PATH = "models/fatigue_cnn_lstm.pth"
SEQUENCE_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "ALERT",
    1: "DROWSY",
    2: "YAWNING"
}

# ======================
# Load model
# ======================
model = CNN_LSTM(
    input_dim=3,     # EAR, MAR, HEAD NOD
    hidden_dim=64,
    num_classes=3
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# Webcam
# ======================
cap = cv2.VideoCapture(0)

sequence = []
prediction = "..."

print("[INFO] Starting webcam fatigue detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ======================
    # Landmark detection
    # ======================
    landmarks = get_landmarks(frame)

    if landmarks is not None:
        ear = compute_ear(landmarks)
        mar = compute_mar(landmarks)
        nod = compute_head_nod(landmarks)

        feature_vec = [ear, mar, nod]
        sequence.append(feature_vec)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # ======================
        # Prediction
        # ======================
        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = torch.tensor(
                np.array(sequence),
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(input_seq)
                pred_class = torch.argmax(outputs, dim=1).item()
                prediction = CLASS_NAMES[pred_class]

    # ======================
    # Display
    # ======================
    cv2.putText(
        frame,
        f"Status: {prediction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255) if prediction != "ALERT" else (0, 255, 0),
        3
    )

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
