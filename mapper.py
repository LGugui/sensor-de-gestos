import ctypes


def _get_screen_size():
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return w, h


class CoordMapper:
    def __init__(self, config):
        self.screen_w, self.screen_h = _get_screen_size()
        self.dead_zone = config["dead_zone_px"]
        self.precision_factor = config["precision_factor"]
        self.alpha = config["ema_alpha"]
        self.margin = config["cam_margin"]
        self.smooth_x = float(self.screen_w // 2)
        self.smooth_y = float(self.screen_h // 2)
        self.prev_x = self.screen_w // 2
        self.prev_y = self.screen_h // 2

    def _normalize(self, v):
        m = self.margin
        return max(0.0, min(1.0, (v - m) / (1.0 - 2 * m)))

    def map(self, landmarks, precision=False):
        lm = landmarks[8]
        nx = self._normalize(lm.x)
        ny = self._normalize(lm.y)

        raw_x = nx * self.screen_w
        raw_y = ny * self.screen_h

        self.smooth_x = self.alpha * raw_x + (1 - self.alpha) * self.smooth_x
        self.smooth_y = self.alpha * raw_y + (1 - self.alpha) * self.smooth_y

        dx = self.smooth_x - self.prev_x
        dy = self.smooth_y - self.prev_y

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
