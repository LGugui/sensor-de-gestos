import ctypes
from collections import deque


def _get_screen_size():
    w = ctypes.windll.user32.GetSystemMetrics(0)
    h = ctypes.windll.user32.GetSystemMetrics(1)
    return w, h


class CoordMapper:
    def __init__(self, config):
        self.screen_w, self.screen_h = _get_screen_size()
        self.dead_zone = config["dead_zone_px"]
        self.precision_factor = config["precision_factor"]
        self.history = deque(maxlen=config["smoothing_frames"])
        self.prev_x = self.screen_w // 2
        self.prev_y = self.screen_h // 2

    def map(self, landmarks, precision=False):
        lm = landmarks[8]
        raw_x = int(lm.x * self.screen_w)
        raw_y = int(lm.y * self.screen_h)

        self.history.append((raw_x, raw_y))
        smooth_x = int(sum(p[0] for p in self.history) / len(self.history))
        smooth_y = int(sum(p[1] for p in self.history) / len(self.history))

        dx = smooth_x - self.prev_x
        dy = smooth_y - self.prev_y

        if precision:
            dx = int(dx * self.precision_factor)
            dy = int(dy * self.precision_factor)

        new_x = max(0, min(self.screen_w - 1, self.prev_x + dx))
        new_y = max(0, min(self.screen_h - 1, self.prev_y + dy))

        if abs(dx) > self.dead_zone or abs(dy) > self.dead_zone:
            self.prev_x = new_x
            self.prev_y = new_y
            return new_x, new_y

        return self.prev_x, self.prev_y
