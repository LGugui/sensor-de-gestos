import math

import numpy as np

from mapper import _OneEuroFilter, _get_screen_size


class FaceMapper:
    """Converts FaceLandmarker outputs to screen coordinates.

    Two cursor modes:
    - Head pose (map_head): yaw/pitch delta from facial_transformation_matrix
    - Gaze (map_gaze): iris position → affine → screen (requires calibration)
    """

    def __init__(self, config):
        self.screen_w, self.screen_h = _get_screen_size()
        self._head_sensitivity  = config.get("head_sensitivity", 2.5)
        self._head_scroll_thr   = config.get("head_scroll_threshold", 0.03)
        self._precision_factor  = config.get("precision_factor", 0.3)
        self._head_x_flip       = config.get("head_x_flip", False)
        self._gaze_x_flip       = config.get("gaze_x_flip", False)
        self._gaze_y_flip       = config.get("gaze_y_flip", False)

        # Head output filters (lighter than gaze — signal is pre-filtered)
        h_cutoff = config.get("head_filter_min_cutoff", 2.0)
        h_beta   = config.get("head_filter_beta", 0.1)
        self._hx = _OneEuroFilter(min_cutoff=h_cutoff, beta=h_beta)
        self._hy = _OneEuroFilter(min_cutoff=h_cutoff, beta=h_beta)

        # Gaze (iris) cursor filters
        g_cutoff = config.get("gaze_filter_min_cutoff", 0.8)
        g_beta   = config.get("gaze_filter_beta", 0.05)
        self._gx = _OneEuroFilter(min_cutoff=g_cutoff, beta=g_beta)
        self._gy = _OneEuroFilter(min_cutoff=g_cutoff, beta=g_beta)

        self.smooth_x = float(self.screen_w // 2)
        self.smooth_y = float(self.screen_h // 2)

        # Signal pre-filters: smooth yaw/pitch BEFORE delta → eliminates flicker
        sig_cutoff = config.get("head_signal_cutoff", 2.0)
        sig_beta   = config.get("head_signal_beta", 0.3)
        self._fyaw   = _OneEuroFilter(min_cutoff=sig_cutoff, beta=sig_beta)
        self._fpitch = _OneEuroFilter(min_cutoff=sig_cutoff, beta=sig_beta)
        self._prev_fyaw   = None
        self._prev_fpitch = None
        self._d_yaw       = 0.0
        self._d_pitch     = 0.0
        self._dead_zone   = config.get("head_dead_zone", 0.001)

        # IPD normalization: midpoint between irises as reference for depth stability
        # When head moves closer (IPD grows), sensitivity scales down proportionally
        self._ref_ipd   = None
        self._ipd_scale = 1.0

        # Blink gate: freeze cursor during blinks to prevent vertical artifacts
        self._blink_gate_thr = config.get("head_blink_gate", 0.4)

        # Absolute pitch tracking for scroll (per-frame delta is too small after filtering)
        self._neutral_pitch = None
        self._last_fpitch   = None

        self._gaze_cal = None
        self._gaze_quality = ""
        self._saccade_passthrough = config.get("saccade_passthrough", False)

    # ── Head pose ─────────────────────────────────────────────────────

    def process_head(self, matrix, face_lm=None, blendshapes=None):
        """Extract yaw/pitch deltas from 4x4 facial transformation matrix.
        face_lm: optional — used for IPD-based sensitivity normalization.
        blendshapes: optional — used to gate cursor during blinks.
        Call exactly once per frame before map_head() / get_head_scroll_delta()."""
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix).reshape(4, 4)

        R = matrix[:3, :3]
        fwd_x = R[0, 2]
        fwd_y = R[1, 2]
        fwd_z = R[2, 2]

        norm_xz = math.sqrt(fwd_x * fwd_x + fwd_z * fwd_z)
        yaw   =  math.atan2(fwd_x, max(fwd_z, 1e-9))
        pitch =  math.atan2(-fwd_y, max(norm_xz, 1e-9))

        # Blink detection: freeze cursor during blink to prevent Y artifacts
        is_blinking = False
        if blendshapes is not None:
            scores = {c.category_name: c.score for c in blendshapes}
            is_blinking = (scores.get("eyeBlinkLeft",  0) > self._blink_gate_thr
                           and scores.get("eyeBlinkRight", 0) > self._blink_gate_thr)

        # Always filter signal to keep filter warm (even during blinks)
        fyaw   = self._fyaw.filter(yaw)
        fpitch = self._fpitch.filter(pitch)

        if self._prev_fyaw is None or is_blinking:
            # First frame OR blink: advance reference without producing delta
            # → no cursor jump after blink completes
            self._prev_fyaw   = fyaw
            self._prev_fpitch = fpitch
            self._d_yaw   = 0.0
            self._d_pitch = 0.0
        else:
            raw_dy = fyaw   - self._prev_fyaw
            raw_dp = fpitch - self._prev_fpitch
            self._d_yaw   = raw_dy if abs(raw_dy) > self._dead_zone else 0.0
            self._d_pitch = raw_dp if abs(raw_dp) > self._dead_zone else 0.0
            self._prev_fyaw   = fyaw
            self._prev_fpitch = fpitch

        # Absolute pitch neutral tracking for scroll detection
        self._last_fpitch = fpitch
        if self._neutral_pitch is None:
            self._neutral_pitch = fpitch
        else:
            dev = fpitch - self._neutral_pitch
            if abs(dev) < self._head_scroll_thr * 0.4 and not is_blinking:
                self._neutral_pitch += dev * 0.04

        # IPD normalization using midpoint between irises (468, 473)
        # Keeps sensitivity stable as head moves toward / away from camera
        if face_lm is not None:
            current_ipd = abs(face_lm[468].x - face_lm[473].x)
            if current_ipd > 0.01:
                if self._ref_ipd is None:
                    self._ref_ipd = current_ipd
                self._ipd_scale = max(0.5, min(2.0, self._ref_ipd / current_ipd))

    def map_head(self, precision=False):
        """Head pose delta → cursor movement. Returns (x, y).
        Eye movement does NOT drive cursor — only head rotation + IPD normalization."""
        flip = -1 if self._head_x_flip else 1

        dx = self._d_yaw   * self._head_sensitivity * self.screen_w * flip * self._ipd_scale
        dy = self._d_pitch * self._head_sensitivity * self.screen_h         * self._ipd_scale

        if precision:
            dx *= self._precision_factor
            dy *= self._precision_factor

        self.smooth_x = max(0.0, min(float(self.screen_w - 1), self.smooth_x + dx))
        self.smooth_y = max(0.0, min(float(self.screen_h - 1), self.smooth_y + dy))

        sx = int(self._hx.filter(self.smooth_x))
        sy = int(self._hy.filter(self.smooth_y))
        return (max(0, min(self.screen_w - 1, sx)),
                max(0, min(self.screen_h - 1, sy)))

    def get_head_scroll_delta(self):
        """Absolute pitch deviation from neutral → scroll ticks. Positive = scroll up."""
        if self._neutral_pitch is None or self._last_fpitch is None:
            return 0
        dev = self._last_fpitch - self._neutral_pitch
        if abs(dev) > self._head_scroll_thr:
            return int(dev * 40)
        return 0

    # ── Gaze (iris) ───────────────────────────────────────────────────

    # Eye boundary landmarks (MediaPipe 478-pt model)
    _R_OUT, _R_IN, _R_TOP, _R_BOT = 33,  133, 159, 145
    _L_OUT, _L_IN, _L_TOP, _L_BOT = 263, 362, 386, 374

    def set_gaze_calibration(self, matrix, quality=""):
        """Arm the affine/polynomial transform. matrix: np.ndarray (2,3) or (2,6)."""
        self._gaze_cal = np.array(matrix)
        self._gaze_quality = quality

    def map_gaze(self, face_lm, saccade=False):
        """Iris → screen coords. Returns (x, y) or None if not calibrated."""
        if self._gaze_cal is None:
            return None
        ix, iy = self.extract_iris_normalized(face_lm)
        if self._gaze_x_flip:
            ix = 1.0 - ix
        if self._gaze_y_flip:
            iy = 1.0 - iy
        if self._gaze_cal.shape[1] == 6:
            v = np.array([ix, iy, ix * ix, iy * iy, ix * iy, 1.0])
        else:
            v = np.array([ix, iy, 1.0])
        x = float(np.dot(self._gaze_cal[0], v))
        y = float(np.dot(self._gaze_cal[1], v))
        if saccade and self._saccade_passthrough:
            sx = int(x)
            sy = int(y)
        else:
            sx = int(self._gx.filter(x))
            sy = int(self._gy.filter(y))
        return (max(0, min(self.screen_w - 1, sx)),
                max(0, min(self.screen_h - 1, sy)))

    def extract_iris_normalized(self, face_lm):
        """Iris position normalized within eye socket [0,1]."""
        r_xl = min(face_lm[self._R_OUT].x, face_lm[self._R_IN].x)
        r_xr = max(face_lm[self._R_OUT].x, face_lm[self._R_IN].x)
        r_yt = min(face_lm[self._R_TOP].y, face_lm[self._R_BOT].y)
        r_yb = max(face_lm[self._R_TOP].y, face_lm[self._R_BOT].y)
        r_nx = (face_lm[468].x - r_xl) / max(r_xr - r_xl, 1e-6)
        r_ny = (face_lm[468].y - r_yt) / max(r_yb - r_yt, 1e-6)

        l_xl = min(face_lm[self._L_OUT].x, face_lm[self._L_IN].x)
        l_xr = max(face_lm[self._L_OUT].x, face_lm[self._L_IN].x)
        l_yt = min(face_lm[self._L_TOP].y, face_lm[self._L_BOT].y)
        l_yb = max(face_lm[self._L_TOP].y, face_lm[self._L_BOT].y)
        l_nx = (face_lm[473].x - l_xl) / max(l_xr - l_xl, 1e-6)
        l_ny = (face_lm[473].y - l_yt) / max(l_yb - l_yt, 1e-6)

        return ((r_nx + l_nx) / 2.0, (r_ny + l_ny) / 2.0)

    def extract_iris_raw(self, face_lm):
        """Raw normalized frame coords (kept for debug/compat)."""
        lx = (face_lm[468].x + face_lm[473].x) / 2.0
        ly = (face_lm[468].y + face_lm[473].y) / 2.0
        return (lx, ly)

    # ── Utilities ─────────────────────────────────────────────────────

    def reset_position(self):
        """Clear filters and return cursor to screen center."""
        self._hx.reset(); self._hy.reset()
        self._gx.reset(); self._gy.reset()
        self._fyaw.reset(); self._fpitch.reset()
        self.smooth_x = float(self.screen_w // 2)
        self.smooth_y = float(self.screen_h // 2)
        self._prev_fyaw   = None
        self._prev_fpitch = None
        self._d_yaw         = 0.0
        self._d_pitch       = 0.0
        self._ref_ipd       = None
        self._ipd_scale     = 1.0
        self._neutral_pitch = None
        self._last_fpitch   = None
