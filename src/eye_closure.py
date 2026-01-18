import numpy as np
from scipy.spatial.distance import euclidean

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, w, h, eye):
    def p(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    A = euclidean(p(eye[1]), p(eye[5]))
    B = euclidean(p(eye[2]), p(eye[4]))
    C = euclidean(p(eye[0]), p(eye[3]))

    return (A + B) / (2.0 * C)

def get_ear(landmarks, w, h):
    left = eye_aspect_ratio(landmarks, w, h, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, w, h, RIGHT_EYE)
    return (left + right) / 2.0
