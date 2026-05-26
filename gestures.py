import math
import time


class GestureDetector:
    def __init__(self, config):
        self.pinch_close = config["pinch_close"]
        self.pinch_open = config["pinch_open"]
        self.precision_min = config["precision_threshold_min"]
        self.precision_max = config["precision_threshold_max"]
        self.scroll_speed = config["scroll_speed"]
        self.menu_hold = config["menu_hold_seconds"]

        self._pinching_left = False
        self._pinching_right = False
        self._prev_scroll_y = None
        self._palm_start = None

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _extended(self, lm, tip, base):
        return lm[tip].y < lm[base].y

    def detect_pinch(self, landmarks):
        dl = self._dist(landmarks[4], landmarks[8])
        dr = self._dist(landmarks[4], landmarks[12])

        fired = None

        # left pinch with hysteresis
        if not self._pinching_left and dl < self.pinch_close:
            self._pinching_left = True
            fired = "left"
        elif self._pinching_left and dl > self.pinch_open:
            self._pinching_left = False

        # right pinch with hysteresis (only if not already left-pinching)
        if not fired:
            if not self._pinching_right and dr < self.pinch_close:
                self._pinching_right = True
                fired = "right"
            elif self._pinching_right and dr > self.pinch_open:
                self._pinching_right = False

        return fired

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

    def detect_open_palm(self, landmarks):
        """All 5 fingers extended. Returns hold progress [0.0, 1.0] or 1.0 when complete."""
        all_extended = (
            self._extended(landmarks, 8, 6)
            and self._extended(landmarks, 12, 10)
            and self._extended(landmarks, 16, 14)
            and self._extended(landmarks, 20, 18)
            and landmarks[4].y < landmarks[3].y  # thumb extended (rough check)
        )

        if all_extended:
            if self._palm_start is None:
                self._palm_start = time.time()
            elapsed = time.time() - self._palm_start
            return min(1.0, elapsed / self.menu_hold)
        else:
            self._palm_start = None
            return 0.0

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

    def reset_pinch(self):
        self._pinching_left = False
        self._pinching_right = False
