import math
import time


class EyeGestures:
    def __init__(self, config):
        self._blink_threshold   = config.get("blink_threshold", 0.5)
        self._dwell_ms          = config.get("dwell_ms", 800) / 1000.0
        self._dwell_radius      = config.get("dwell_radius_px", 40)
        self._saccade_threshold = config.get("saccade_threshold", 0.15)
        self._double_blink_ms   = config.get("double_blink_ms", 400) / 1000.0
        self._long_close_s      = config.get("keyboard_long_close_s", 4.0)
        self._wink_scroll_s     = config.get("wink_scroll_s", 3.0)
        self._wink_open_thr     = config.get("wink_open_threshold", 0.25)

        self._prev_blink       = False
        self._last_blink_t     = None
        self._long_close_start = None
        self._left_wink_start  = None

        self._dwell_anchor_x = None
        self._dwell_anchor_y = None
        self._dwell_start    = None
        self._prev_iris_x    = None
        self._prev_iris_y    = None
        self._prev_iris_t    = None

    # ── Basic blink ───────────────────────────────────────────────────

    def detect_blink(self, blendshapes):
        """Both eyes close simultaneously → rising edge True."""
        scores = {cat.category_name: cat.score for cat in blendshapes}
        is_blink = (
            scores.get("eyeBlinkLeft",  0) > self._blink_threshold
            and scores.get("eyeBlinkRight", 0) > self._blink_threshold
        )
        fired = is_blink and not self._prev_blink
        self._prev_blink = is_blink
        return fired

    def detect_double_blink(self, blendshapes):
        """Two blinks within double_blink_ms window → True. Single blink ignored."""
        fired = self.detect_blink(blendshapes)
        if fired:
            now = time.time()
            if self._last_blink_t is not None and (now - self._last_blink_t) < self._double_blink_ms:
                self._last_blink_t = None
                return True
            self._last_blink_t = now
        return False

    # ── Sustained gestures ────────────────────────────────────────────

    def detect_left_wink(self, blendshapes):
        """Left eye closed + right eye open for wink_scroll_s → True once."""
        scores = {cat.category_name: cat.score for cat in blendshapes}
        left_closed = scores.get("eyeBlinkLeft",  0) > self._blink_threshold
        right_open  = scores.get("eyeBlinkRight", 0) < self._wink_open_thr
        winking = left_closed and right_open
        if winking:
            if self._left_wink_start is None:
                self._left_wink_start = time.time()
            elif time.time() - self._left_wink_start >= self._wink_scroll_s:
                self._left_wink_start = None
                return True
        else:
            self._left_wink_start = None
        return False

    def detect_long_close(self, blendshapes):
        """Both eyes held closed for long_close_s → True once. Resets on open."""
        scores = {cat.category_name: cat.score for cat in blendshapes}
        both_closed = (scores.get("eyeBlinkLeft",  0) > self._blink_threshold
                       and scores.get("eyeBlinkRight", 0) > self._blink_threshold)
        if both_closed:
            if self._long_close_start is None:
                self._long_close_start = time.time()
            elif time.time() - self._long_close_start >= self._long_close_s:
                self._long_close_start = None
                return True
        else:
            self._long_close_start = None
        return False

    # ── Progress properties (for overlay display) ─────────────────────

    @property
    def long_close_progress(self):
        if self._long_close_start is None:
            return 0.0
        return min(1.0, (time.time() - self._long_close_start) / self._long_close_s)

    @property
    def wink_progress(self):
        if self._left_wink_start is None:
            return 0.0
        return min(1.0, (time.time() - self._left_wink_start) / self._wink_scroll_s)

    # ── Iris / gaze ───────────────────────────────────────────────────

    def is_saccade(self, ix, iy):
        """True if normalized iris velocity exceeds saccade_threshold (units/s)."""
        now = time.time()
        if self._prev_iris_x is None:
            self._prev_iris_x = ix
            self._prev_iris_y = iy
            self._prev_iris_t = now
            return False
        dt  = max(now - self._prev_iris_t, 0.001)
        vel = math.sqrt((ix - self._prev_iris_x) ** 2 + (iy - self._prev_iris_y) ** 2) / dt
        self._prev_iris_x = ix
        self._prev_iris_y = iy
        self._prev_iris_t = now
        return vel > self._saccade_threshold

    @property
    def dwell_progress(self):
        if self._dwell_anchor_x is None or self._dwell_start is None:
            return 0.0
        return min(1.0, (time.time() - self._dwell_start) / self._dwell_ms)

    def update_dwell(self, x, y):
        """Gaze stable within radius for dwell_ms → returns True once per dwell."""
        if self._dwell_anchor_x is None:
            self._dwell_anchor_x = x
            self._dwell_anchor_y = y
            self._dwell_start = time.time()
            return False

        dist = ((x - self._dwell_anchor_x) ** 2 + (y - self._dwell_anchor_y) ** 2) ** 0.5
        if dist > self._dwell_radius:
            self._dwell_anchor_x = x
            self._dwell_anchor_y = y
            self._dwell_start = time.time()
            return False

        if time.time() - self._dwell_start >= self._dwell_ms:
            self._dwell_start = time.time()
            return True

        return False
