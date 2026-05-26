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
        self._dual_palm_start = None
        self._fist_prev = False
        self._victory_start = None

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _extended(self, lm, tip, base):
        return lm[tip].y < lm[base].y

    def _all_extended(self, lm):
        return (
            self._extended(lm, 8, 6)
            and self._extended(lm, 12, 10)
            and self._extended(lm, 16, 14)
            and self._extended(lm, 20, 18)
            and lm[4].y < lm[3].y
        )

    def _all_closed(self, lm):
        return (
            not self._extended(lm, 8, 6)
            and not self._extended(lm, 12, 10)
            and not self._extended(lm, 16, 14)
            and not self._extended(lm, 20, 18)
        )

    # ── Cursor hand gestures ─────────────────────────────────────────────

    def detect_pinch(self, landmarks):
        """Returns 'left', 'right' on transition, else None. Has hysteresis."""
        dl = self._dist(landmarks[4], landmarks[8])
        dr = self._dist(landmarks[4], landmarks[12])
        fired = None

        if not self._pinching_left and dl < self.pinch_close:
            self._pinching_left = True
            fired = "left"
        elif self._pinching_left and dl > self.pinch_open:
            self._pinching_left = False

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

    # ── Control hand gestures ─────────────────────────────────────────────

    def detect_fist(self, landmarks):
        """Returns True on the frame the fist closes (rising edge)."""
        is_fist = self._all_closed(landmarks)
        fired = is_fist and not self._fist_prev
        self._fist_prev = is_fist
        return fired

    # ── Two-hand gestures ─────────────────────────────────────────────────

    def detect_open_palm(self, landmarks):
        """Single-hand palm hold. Returns progress [0,1]. Used as fallback."""
        if self._all_extended(landmarks):
            if self._palm_start is None:
                self._palm_start = time.time()
            return min(1.0, (time.time() - self._palm_start) / self.menu_hold)
        self._palm_start = None
        return 0.0

    def detect_victory(self, landmarks):
        """V sign: index + middle extended, others closed. Returns hold progress [0,1]."""
        is_v = (
            self._extended(landmarks, 8, 6)
            and self._extended(landmarks, 12, 10)
            and not self._extended(landmarks, 16, 14)
            and not self._extended(landmarks, 20, 18)
            and not (landmarks[4].y < landmarks[3].y)  # thumb NOT extended
        )
        if is_v:
            if self._victory_start is None:
                self._victory_start = time.time()
            return min(1.0, (time.time() - self._victory_start) / self.menu_hold)
        self._victory_start = None
        return 0.0

    def detect_dual_palm(self, lm_a, lm_b):
        """Both palms open simultaneously. Returns progress [0,1]."""
        if self._all_extended(lm_a) and self._all_extended(lm_b):
            if self._dual_palm_start is None:
                self._dual_palm_start = time.time()
            return min(1.0, (time.time() - self._dual_palm_start) / self.menu_hold)
        self._dual_palm_start = None
        return 0.0
