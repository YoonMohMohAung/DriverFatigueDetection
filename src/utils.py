import numpy as np
from scipy.spatial import distance as dist

# =========================
# PERCLOS
# =========================
class Perclos:
    def __init__(self):
        self.total_frames = 0
        self.closed_frames = 0

    def update(self, ear, threshold=0.25):
        self.total_frames += 1
        if ear < threshold:
            self.closed_frames += 1

    def value(self):
        if self.total_frames == 0:
            return 0.0
        return self.closed_frames / self.total_frames


# =========================
# Helper
# =========================
def _pt(lm):
    """Convert MediaPipe landmark to numpy (x, y)"""
    return np.array([lm.x, lm.y])


# =========================
# Eye Aspect Ratio (EAR)
# =========================
def compute_ear(landmarks):
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    def ear(eye):
        A = dist.euclidean(_pt(landmarks[eye[1]]), _pt(landmarks[eye[5]]))
        B = dist.euclidean(_pt(landmarks[eye[2]]), _pt(landmarks[eye[4]]))
        C = dist.euclidean(_pt(landmarks[eye[0]]), _pt(landmarks[eye[3]]))
        return (A + B) / (2.0 * C)

    return (ear(left_eye) + ear(right_eye)) / 2.0


# =========================
# Mouth Aspect Ratio (MAR)
# =========================
def compute_mar(landmarks):
    top = _pt(landmarks[13])
    bottom = _pt(landmarks[14])
    left = _pt(landmarks[78])
    right = _pt(landmarks[308])

    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)

    return vertical / horizontal


# =========================
# Head Nod Metric
# =========================
def compute_head_nod(landmarks):
    nose_tip = _pt(landmarks[1])
    chin = _pt(landmarks[152])

    return dist.euclidean(nose_tip, chin)
