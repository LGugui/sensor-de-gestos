import ctypes
import math
import time


def _get_screen_size():
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return w, h


class _OneEuroFilter:
    """Adaptive low-pass filter: heavy smoothing at rest, light smoothing during fast movement.
    Eliminates tremor without adding lag. Ref: Casiez et al. 2012."""

    def __init__(self, min_cutoff=0.5, beta=0.1, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = 0.0
        self._t = None

    def _alpha(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def filter(self, x):
        now = time.perf_counter()
        if self._x is None:
            self._x = x
            self._t = now
            return x
        te = now - self._t
        if te <= 0:
            return self._x
        self._t = now
        # Filtered derivative
        raw_dx = (x - self._x) / te
        a_d = self._alpha(te, self.d_cutoff)
        self._dx = a_d * raw_dx + (1 - a_d) * self._dx
        # Adaptive cutoff: higher speed → less smoothing → less lag
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(te, cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._t = None


class CoordMapper:
    def __init__(self, config):
        self.screen_w, self.screen_h = _get_screen_size()
        self.dead_zone = config["dead_zone_px"]
        self.precision_factor = config["precision_factor"]
        self.margin = config["cam_margin"]
        self.mode = config.get("mapping_mode", "absolute")
        self.sensitivity = config.get("sensitivity", 1.5)
        self.acceleration = config.get("acceleration", True)
        self.accel_threshold = config.get("accel_threshold", 8)
        self.accel_factor = config.get("accel_factor", 0.08)
        # lm[9] = middle MCP (palm center) — far more stable than lm[8] (index tip)
        self._lm_idx = config.get("cursor_landmark", 9)

        min_cutoff = config.get("filter_min_cutoff", 0.5)
        beta = config.get("filter_beta", 0.1)
        self._fx = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._fy = _OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        self.smooth_x = float(self.screen_w // 2)
        self.smooth_y = float(self.screen_h // 2)
        self._prev_lm_x = None
        self._prev_lm_y = None
        self.prev_x = self.screen_w // 2
        self.prev_y = self.screen_h // 2
        self._cal = None        # (min_x, max_x, min_y, max_y) — set by calibrate()
        self._ar_bounds = None  # computed from screen/camera aspect ratio

    def calibrate(self, min_x, max_x, min_y, max_y):
        self._cal = (min_x, max_x, min_y, max_y)

    def set_frame_size(self, cam_w, cam_h):
        """Compute mapping bounds so active zone matches screen aspect ratio."""
        ar_s = self.screen_w / self.screen_h
        ar_c = cam_w / cam_h
        if ar_s > ar_c:   # screen wider → crop top/bottom
            my = (1.0 - ar_c / ar_s) / 2
            mx = 0.0
        else:              # screen taller → crop sides
            mx = (1.0 - ar_s / ar_c) / 2
            my = 0.0
        self._ar_bounds = (mx, 1.0 - mx, my, 1.0 - my)

    @property
    def effective_bounds(self):
        """Active mapping area (min_x, max_x, min_y, max_y) in normalized coords."""
        if self._cal:
            return self._cal
        if self._ar_bounds:
            return self._ar_bounds
        m = self.margin
        return (m, 1.0 - m, m, 1.0 - m)

    def _norm_cal(self, v, lo, hi):
        if hi - lo < 0.01:
            return 0.5
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    def reset_position(self):
        """Call when cursor hand is lost — prevents jump on re-detection."""
        self._prev_lm_x = None
        self._prev_lm_y = None
        self._fx.reset()
        self._fy.reset()

    def _in_active_area(self, tip):
        m = self.margin
        return m <= tip.x <= 1 - m and m <= tip.y <= 1 - m

    def _normalize(self, v):
        m = self.margin
        return max(0.0, min(1.0, (v - m) / (1.0 - 2 * m)))

    def map(self, landmarks, precision=False, depth=None):
        tip = landmarks[self._lm_idx]
        if self.mode == "relative":
            return self._map_relative(tip, precision, depth)
        return self._map_absolute(tip, precision, depth)

    def map_draw(self, landmarks, precision=False):
        """Draw mode: always uses index tip (lm[8]) — pen follows fingertip."""
        tip = landmarks[8]
        if self.mode == "relative":
            return self._map_relative(tip, precision)
        return self._map_absolute(tip, precision)

    def _map_absolute(self, tip, precision, depth=None):
        b = self.effective_bounds
        nx = self._norm_cal(tip.x, b[0], b[1])
        ny = self._norm_cal(tip.y, b[2], b[3])
        raw_x = nx * self.screen_w
        raw_y = ny * self.screen_h

        # One Euro Filter removes tremor adaptively
        sx = self._fx.filter(raw_x)
        sy = self._fy.filter(raw_y)

        dx = sx - self.prev_x
        dy = sy - self.prev_y

        # Auto-precision: hand close to camera → precision mode
        if depth is not None and depth > 0.20:
            precision = True

        if precision:
            dx *= self.precision_factor
            dy *= self.precision_factor

        new_x = max(0, min(self.screen_w - 1, int(self.prev_x + dx)))
        new_y = max(0, min(self.screen_h - 1, int(self.prev_y + dy)))

        if abs(dx) > self.dead_zone or abs(dy) > self.dead_zone:
            self.prev_x = new_x
            self.prev_y = new_y
            return new_x, new_y
        return self.prev_x, self.prev_y

    def _map_relative(self, tip, precision, depth=None):
        if self._prev_lm_x is None:
            self._prev_lm_x = tip.x
            self._prev_lm_y = tip.y
            return int(self.smooth_x), int(self.smooth_y)

        if not self._in_active_area(tip):
            # Track position so no jump when re-entering active area
            self._prev_lm_x = tip.x
            self._prev_lm_y = tip.y
            return int(self.smooth_x), int(self.smooth_y)

        dx = (tip.x - self._prev_lm_x) * self.screen_w * self.sensitivity
        dy = (tip.y - self._prev_lm_y) * self.screen_h * self.sensitivity
        self._prev_lm_x = tip.x
        self._prev_lm_y = tip.y

        # Depth-adaptive sensitivity: far (small hand) = faster, close (large) = slower
        if depth is not None:
            depth_scale = max(0.4, min(2.5, 0.15 / max(0.05, depth)))
            dx *= depth_scale
            dy *= depth_scale

        if self.acceleration:
            speed = math.sqrt(dx * dx + dy * dy)
            if speed > self.accel_threshold:
                factor = 1.0 + (speed - self.accel_threshold) * self.accel_factor
                dx *= factor
                dy *= factor

        if precision:
            dx *= self.precision_factor
            dy *= self.precision_factor

        self.smooth_x = max(0.0, min(float(self.screen_w - 1), self.smooth_x + dx))
        self.smooth_y = max(0.0, min(float(self.screen_h - 1), self.smooth_y + dy))
        return int(self.smooth_x), int(self.smooth_y)
