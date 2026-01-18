class HeadNodDetector:
    def __init__(self):
        self.prev_y = None
        self.nod_count = 0

    def update(self, landmarks, h, threshold=15):
        nose_y = landmarks[1].y * h

        if self.prev_y is None:
            self.prev_y = nose_y
            return

        delta = nose_y - self.prev_y

        if delta > threshold:
            self.nod_count += 1

        self.prev_y = nose_y
