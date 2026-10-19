import json
import os
import time

import numpy as np

from mapper import _get_screen_size

_CAL_PATH = os.path.join(os.path.dirname(__file__), "gaze_calibration.json")
_CAL_VERSION = 4   # v4 = polynomial (2,6); v3 = affine (2,3); v1/v2 rejected

HOLD_SECS = 2.0
_BLINK_REJECT_THR = 0.35
_EARLY_STOP_VAR   = 0.0003
_TRIM_FRAC        = 0.15

# 9-point 3x3 grid — full coverage of screen, especially Y axis
_POINTS = [
    ("CENTRO",          0.50, 0.50),
    ("SUPERIOR ESQ",    0.15, 0.10),
    ("SUPERIOR CENTRO", 0.50, 0.10),
    ("SUPERIOR DIR",    0.85, 0.10),
    ("MEIO ESQ",        0.15, 0.50),
    ("MEIO DIR",        0.85, 0.50),
    ("INFERIOR ESQ",    0.15, 0.90),
    ("INFERIOR CENTRO", 0.50, 0.90),
    ("INFERIOR DIR",    0.85, 0.90),
]


def _trim_mean(values):
    n = len(values)
    if n < 4:
        return sum(values) / n
    cut = max(1, int(n * _TRIM_FRAC))
    s = sorted(values)
    trimmed = s[cut:n - cut]
    return sum(trimmed) / len(trimmed)


class GazeCalibrator:
    """9-point gaze calibration — polynomial regression, blink rejection, trimmed mean."""

    def __init__(self):
        self._step = 0
        self._hold_start = None
        self._samples = []
        self._data = []
        self._targets = []
        self.done = False
        self.result = None      # np.ndarray shape (2,6) — polynomial coefficients
        self.quality = ""       # "boa" | "razoavel" | "ruim"
        self.residual_px = 0.0

        sw, sh = _get_screen_size()
        self._screen_w = sw
        self._screen_h = sh

    @property
    def label(self):
        return _POINTS[self._step][0] if self._step < len(_POINTS) else ""

    @property
    def target_fx(self):
        return _POINTS[self._step][1] if self._step < len(_POINTS) else 0

    @property
    def target_fy(self):
        return _POINTS[self._step][2] if self._step < len(_POINTS) else 0

    @property
    def step(self):
        return self._step

    @property
    def total(self):
        return len(_POINTS)

    def update(self, face_lm, face_mapper, face_blendshapes=None):
        """Accumulate normalized iris samples. Returns hold progress [0,1]."""
        if self.done or self._step >= len(_POINTS):
            return None

        if self._hold_start is None:
            self._hold_start = time.time()

        # Skip sample when eyes are closing (blink rejection)
        if face_blendshapes is not None:
            bs = {b.category_name: b.score for b in face_blendshapes}
            if (bs.get("eyeBlinkLeft",  0) > _BLINK_REJECT_THR or
                    bs.get("eyeBlinkRight", 0) > _BLINK_REJECT_THR):
                return min(1.0, (time.time() - self._hold_start) / HOLD_SECS)

        ix, iy = face_mapper.extract_iris_normalized(face_lm)
        self._samples.append((ix, iy))

        progress = min(1.0, (time.time() - self._hold_start) / HOLD_SECS)

        # Early stop: gaze is stable — no need to wait full HOLD_SECS
        if len(self._samples) >= 20:
            recent = self._samples[-20:]
            xs = [s[0] for s in recent]
            ys = [s[1] for s in recent]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            var_x = sum((x - mx) ** 2 for x in xs) / len(xs)
            var_y = sum((y - my) ** 2 for y in ys) / len(ys)
            if var_x < _EARLY_STOP_VAR and var_y < _EARLY_STOP_VAR:
                progress = 1.0

        if progress >= 1.0:
            xs = [s[0] for s in self._samples]
            ys = [s[1] for s in self._samples]
            self._data.append((_trim_mean(xs), _trim_mean(ys)))

            _, tx, ty = _POINTS[self._step]
            self._targets.append((tx * self._screen_w, ty * self._screen_h))

            self._step += 1
            self._hold_start = None
            self._samples = []

            if self._step >= len(_POINTS):
                self._finalize()

        return progress

    @staticmethod
    def _features(ix, iy):
        return [ix, iy, ix * ix, iy * iy, ix * iy, 1.0]

    def _finalize(self):
        A  = np.array([self._features(ix, iy) for ix, iy in self._data])  # (9, 6)
        bx = np.array([t[0] for t in self._targets])
        by = np.array([t[1] for t in self._targets])
        cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        self.result = np.stack([cx, cy])  # (2, 6)

        pred_x = A @ cx
        pred_y = A @ cy
        residuals = np.sqrt((pred_x - bx) ** 2 + (pred_y - by) ** 2)
        self.residual_px = float(np.sqrt(np.mean(residuals ** 2)))

        if self.residual_px < 40:
            self.quality = "boa"
        elif self.residual_px < 80:
            self.quality = "razoavel"
        else:
            self.quality = "ruim"

        self.done = True


def load_gaze_calibration():
    """Returns np.ndarray (2,N) or None. Accepts v3 (affine) and v4 (polynomial)."""
    if not os.path.exists(_CAL_PATH):
        return None
    try:
        with open(_CAL_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("v") not in (3, _CAL_VERSION):
            return None
        return np.array(data["matrix"])
    except Exception:
        return None


def save_gaze_calibration(matrix, quality="", residual_px=0.0):
    with open(_CAL_PATH, "w") as f:
        json.dump({
            "v": _CAL_VERSION,
            "matrix": matrix.tolist(),
            "quality": quality,
            "residual_px": float(residual_px),
        }, f)
