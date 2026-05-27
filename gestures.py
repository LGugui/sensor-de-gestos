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

        self.draw_hold = config.get("draw_hold_seconds", 0.8)

        self._pinching_left = False
        self._pinching_right = False
        self._pinch_left_frames = 0
        self._pinch_right_frames = 0
        self._pinch_frames_required = config.get("pinch_frames_required", 2)
        self._prev_scroll_y = None
        self._palm_start = None
        self._dual_palm_start = None
        self.touch_dist = config.get("touch_dist", 0.07)

        self._fist_prev = False
        self._pinky_prev = False
        self._touch_prev = False
        self._touch_span = 0.2
        self._victory_start = None
        self._last_click_t = None
        self._prev_depth = None

        # TF classifier (optional — falls back to rule-based if absent)
        self._tf = None
        self._tf_cache_key = None
        self._tf_cache_probs = {}
        try:
            from tf_gesture import TFGestureClassifier
            self._tf = TFGestureClassifier()
        except Exception:
            pass  # no model or TF not installed — rules only

    # ── TF inference (per-frame cache) ───────────────────────────────────

    def _tf_probs(self, lm):
        """Returns TF probability dict for lm, cached per unique frame."""
        if self._tf is None:
            return {}
        # Cache key: wrist position rounded to 4 decimals (changes every frame when moving)
        key = (round(lm[0].x, 4), round(lm[0].y, 4))
        if self._tf_cache_key != key:
            self._tf_cache_probs = self._tf.classify(lm)
            self._tf_cache_key = key
        return self._tf_cache_probs

    # ── Pseudo-depth (webcam only) ────────────────────────────────────────
    # Wrist-to-middle-MCP distance in normalized space correlates with
    # hand distance from camera: ~0.10 = far (~80cm), ~0.15 = normal (~50cm),
    # ~0.25 = close (~25cm).
    _DEPTH_REF = 0.15

    def estimate_depth(self, lm):
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        return math.sqrt(dx * dx + dy * dy)

    def get_depth_scroll_delta(self, lm):
        """Scroll delta from hand moving toward/away camera."""
        depth = self.estimate_depth(lm)
        if self._prev_depth is None:
            self._prev_depth = depth
            return 0
        delta = (depth - self._prev_depth) * self.scroll_speed * 60
        self._prev_depth = depth
        return int(delta)

    def reset_depth(self):
        self._prev_depth = None

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _extended_sq(self, lm, tip, base):
        """Tip farther from wrist than base → extended. Uses squared distance (no sqrt)."""
        w = lm[0]
        dt2 = (lm[tip].x - w.x)**2 + (lm[tip].y - w.y)**2 + (lm[tip].z - w.z)**2
        db2 = (lm[base].x - w.x)**2 + (lm[base].y - w.y)**2 + (lm[base].z - w.z)**2
        return dt2 > db2

    def finger_states(self, lm):
        return {
            "thumb":  self._extended_sq(lm, 4, 2),
            "index":  self._extended_sq(lm, 8, 6),
            "middle": self._extended_sq(lm, 12, 10),
            "ring":   self._extended_sq(lm, 16, 14),
            "pinky":  self._extended_sq(lm, 20, 18),
        }

    def _all_extended(self, lm):
        return (
            self._extended_sq(lm, 8, 6)
            and self._extended_sq(lm, 12, 10)
            and self._extended_sq(lm, 16, 14)
            and self._extended_sq(lm, 20, 18)
            and self._extended_sq(lm, 4, 2)
        )

    def _all_closed(self, lm):
        return (
            not self._extended_sq(lm, 8, 6)
            and not self._extended_sq(lm, 12, 10)
            and not self._extended_sq(lm, 16, 14)
            and not self._extended_sq(lm, 20, 18)
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_pinching(self):
        return self._pinching_left or self._pinching_right

    @property
    def is_dragging(self):
        return self._pinching_left

    # ── Cursor hand gestures ─────────────────────────────────────────────

    def detect_double_click(self, landmarks):
        """Returns True on the rising edge of a second pinch within 400ms.
        Must be called BEFORE detect_pinch — consumes the second pinch event."""
        if self._pinching_left:
            return False
        probs = self._tf_probs(landmarks)
        is_pinch = (probs.get("pinch_left", 0) > 0.70) if probs else (
            self._dist(landmarks[4], landmarks[8]) < self.pinch_close)
        if is_pinch:
            now = time.time()
            if self._last_click_t and now - self._last_click_t < 0.4:
                self._last_click_t = None
                self._pinching_left = True  # consume so detect_pinch won't fire
                return True
            self._last_click_t = now
        return False

    def detect_pinch(self, landmarks):
        """Returns 'left', 'right' on transition, else None.
        Requires pinch_frames_required consecutive frames to prevent false positives."""
        probs = self._tf_probs(landmarks)
        fired = None

        if probs:
            is_l = (probs.get("pinch_left", 0) > (0.35 if self._pinching_left else 0.70))
            is_r = (probs.get("pinch_right", 0) > (0.35 if self._pinching_right else 0.70))
        else:
            dl = self._dist(landmarks[4], landmarks[8])
            dr = self._dist(landmarks[4], landmarks[12])
            is_l = dl < (self.pinch_open if self._pinching_left else self.pinch_close)
            is_r = dr < (self.pinch_open if self._pinching_right else self.pinch_close)

        # Left pinch with frame debounce
        if is_l:
            self._pinch_left_frames += 1
        else:
            self._pinch_left_frames = 0

        if not self._pinching_left and self._pinch_left_frames >= self._pinch_frames_required:
            self._pinching_left = True
            fired = "left"
        elif self._pinching_left and not is_l:
            self._pinching_left = False

        # Right pinch with frame debounce
        if is_r:
            self._pinch_right_frames += 1
        else:
            self._pinch_right_frames = 0

        if not fired:
            if not self._pinching_right and self._pinch_right_frames >= self._pinch_frames_required:
                self._pinching_right = True
                fired = "right"
            elif self._pinching_right and not is_r:
                self._pinching_right = False

        return fired

    def detect_precision(self, landmarks):
        probs = self._tf_probs(landmarks)
        if probs:
            return probs.get("precision", 0) > 0.60
        d = self._dist(landmarks[4], landmarks[8])
        return self.precision_min < d < self.precision_max

    def detect_scroll_mode(self, landmarks):
        probs = self._tf_probs(landmarks)
        if probs:
            return probs.get("scroll", 0) > 0.65
        return (
            self._extended_sq(landmarks, 8, 6)
            and self._extended_sq(landmarks, 12, 10)
            and not self._extended_sq(landmarks, 16, 14)
            and not self._extended_sq(landmarks, 20, 18)
        )

    def get_scroll_delta(self, landmarks):
        # Use average of index+middle tips (the scroll gesture fingers) — more stable than wrist
        curr_y = (landmarks[8].y + landmarks[12].y) / 2
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
        self._pinch_left_frames = 0
        self._pinch_right_frames = 0

    def reset_victory(self):
        self._victory_start = None

    # ── Control hand gestures ─────────────────────────────────────────────

    def detect_fist(self, landmarks):
        """Returns True on the frame the fist closes (rising edge)."""
        probs = self._tf_probs(landmarks)
        is_fist = (probs.get("fist", 0) > 0.70) if probs else self._all_closed(landmarks)
        fired = is_fist and not self._fist_prev
        self._fist_prev = is_fist
        return fired

    def detect_pinky_only(self, landmarks):
        probs = self._tf_probs(landmarks)
        if probs:
            is_pinky = probs.get("pinky_only", 0) > 0.70
        else:
            is_pinky = (
                self._extended_sq(landmarks, 20, 18)
                and not self._extended_sq(landmarks, 8, 6)
                and not self._extended_sq(landmarks, 12, 10)
                and not self._extended_sq(landmarks, 16, 14)
            )
        fired = is_pinky and not self._pinky_prev
        self._pinky_prev = is_pinky
        return fired

    # ── Two-hand gestures ─────────────────────────────────────────────────

    def detect_open_palm(self, landmarks):
        probs = self._tf_probs(landmarks)
        is_palm = (probs.get("open_palm", 0) > 0.65) if probs else self._all_extended(landmarks)
        if is_palm:
            if self._palm_start is None:
                self._palm_start = time.time()
            return min(1.0, (time.time() - self._palm_start) / self.menu_hold)
        self._palm_start = None
        return 0.0

    def detect_ring_hold(self, landmarks):
        """Ring finger only extended, others closed. Returns hold progress [0,1]."""
        probs = self._tf_probs(landmarks)
        if probs:
            is_ring = probs.get("ring_only", 0) > 0.65
        else:
            is_ring = (
                self._extended_sq(landmarks, 16, 14)
                and not self._extended_sq(landmarks, 8, 6)
                and not self._extended_sq(landmarks, 12, 10)
                and not self._extended_sq(landmarks, 20, 18)
            )
        if is_ring:
            if self._victory_start is None:
                self._victory_start = time.time()
            return min(1.0, (time.time() - self._victory_start) / self.draw_hold)
        self._victory_start = None
        return 0.0

    def detect_dual_palm(self, lm_a, lm_b):
        if self._all_extended(lm_a) and self._all_extended(lm_b):
            if self._dual_palm_start is None:
                self._dual_palm_start = time.time()
            return min(1.0, (time.time() - self._dual_palm_start) / self.menu_hold)
        self._dual_palm_start = None
        return 0.0

    def detect_hand_touch(self, lm_a, lm_b):
        """Index tips from both hands touch. Rising edge only."""
        d_tips = self._dist(lm_a[8], lm_b[8])
        is_touching = d_tips < self.touch_dist
        if not is_touching:
            self._touch_span = d_tips
        fired = is_touching and not self._touch_prev
        self._touch_prev = is_touching
        if fired:
            mx = (lm_a[8].x + lm_b[8].x) / 2
            my = (lm_a[8].y + lm_b[8].y) / 2
            return True, mx, my, self._touch_span
        return False, None, None, 0.0
