import numpy as np
from scipy.spatial.distance import euclidean

# Mouth landmarks (MediaPipe)
UPPER_LIP = [13]
LOWER_LIP = [14]
LEFT_MOUTH = [78]
RIGHT_MOUTH = [308]

def mouth_aspect_ratio(landmarks, w, h):
    def p(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    vertical = euclidean(p(UPPER_LIP[0]), p(LOWER_LIP[0]))
    horizontal = euclidean(p(LEFT_MOUTH[0]), p(RIGHT_MOUTH[0]))

    mar = vertical / horizontal
    return mar
