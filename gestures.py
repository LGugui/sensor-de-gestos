import math


class GestureDetector:
    def __init__(self, config):
        self.pinch_threshold = config["pinch_threshold"]
        self.precision_min = config["precision_threshold_min"]
        self.precision_max = config["precision_threshold_max"]
        self.scroll_speed = config["scroll_speed"]
        self._prev_scroll_y = None

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _extended(self, lm, tip, base):
        return lm[tip].y < lm[base].y

    def detect_pinch(self, landmarks):
        if self._dist(landmarks[4], landmarks[8]) < self.pinch_threshold:
            return "left"
        if self._dist(landmarks[4], landmarks[12]) < self.pinch_threshold:
            return "right"
        return None

    def detect_precision(self, landmarks):
        d = self._dist(landmarks[4], landmarks[8])
        return self.precision_min < d < self.precision_max

    def detect_scroll_mode(self, landmarks):
        return (
            self._extended(landmarks, 8, 6)
            and self._extended(landmarks, 12, 10)
            and not self._extended(landmarks, 16, 14)
            and not self._extended(landmarks, 20, 18)
        )

    def get_scroll_delta(self, landmarks):
        curr_y = landmarks[0].y
        if self._prev_scroll_y is None:
            self._prev_scroll_y = curr_y
            return 0
        delta = (self._prev_scroll_y - curr_y) * self.scroll_speed * 10
        self._prev_scroll_y = curr_y
        return int(delta)

    def reset_scroll(self):
        self._prev_scroll_y = None
