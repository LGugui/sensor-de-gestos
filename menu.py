import config as cfg_module
from config import DEFAULTS


def _clamp_alpha(cfg, delta):
    cfg["ema_alpha"] = round(max(0.1, min(0.9, cfg["ema_alpha"] + delta)), 2)


def _clamp_scroll(cfg, delta):
    cfg["scroll_speed"] = max(1, cfg["scroll_speed"] + delta)


ITEMS = [
    ("Sensibilidade +", lambda cfg: _clamp_alpha(cfg, +0.05)),
    ("Sensibilidade -", lambda cfg: _clamp_alpha(cfg, -0.05)),
    ("Scroll +",        lambda cfg: _clamp_scroll(cfg, +1)),
    ("Scroll -",        lambda cfg: _clamp_scroll(cfg, -1)),
    ("Reset Config",    lambda cfg: cfg.update(DEFAULTS)),
    ("Fechar Menu",     None),
    ("Fechar App",      "exit"),
]

ITEM_H = 44
ITEM_W = 220
PADDING = 12


class GestureMenu:
    def __init__(self):
        self.open = False
        self.hovered = -1
        self._x0 = 0
        self._y0 = 0

    def toggle(self):
        self.open = not self.open
        self.hovered = -1

    def close(self):
        self.open = False
        self.hovered = -1

    def update_hover(self, frame_w, frame_h, cursor_screen_x, cursor_screen_y, screen_w, screen_h):
        """Map screen cursor position back to frame coordinates for hover detection."""
        fx = int(cursor_screen_x / screen_w * frame_w)
        fy = int(cursor_screen_y / screen_h * frame_h)
        self.hovered = -1
        for i in range(len(ITEMS)):
            ix = self._x0
            iy = self._y0 + i * ITEM_H
            if ix <= fx <= ix + ITEM_W and iy <= fy <= iy + ITEM_H:
                self.hovered = i
                break

    def select(self, cfg):
        """Execute hovered item action. Returns 'exit', 'close', or None."""
        i = self.hovered
        if i < 0 or i >= len(ITEMS):
            return "close"
        _, action = ITEMS[i]
        if action == "exit":
            return "exit"
        if action is None:
            self.close()
            return "close"
        action(cfg)
        cfg_module.save(cfg)
        self.close()
        return "close"

    def anchor(self, frame_w, frame_h):
        """Compute top-left of menu panel (center of frame)."""
        total_h = len(ITEMS) * ITEM_H + PADDING * 2
        self._x0 = (frame_w - ITEM_W) // 2
        self._y0 = (frame_h - total_h) // 2 + PADDING
        return self._x0, self._y0
