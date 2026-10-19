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

# Fan menu geometry
CARD_W = 115
CARD_H = 32
_FAN_MCP_INDICES = [5, 9, 13, 17]   # index, middle, ring, pinky MCPs
_FAN_OFFSET_Y = 60                   # pixels above MCP for card center


class GestureMenu:
    def __init__(self):
        self.open = False
        self.hovered = -1
        self._x0 = 0
        self._y0 = 0
        self.fan_positions = []      # list of (cx, cy) per item, set by anchor_to_hand

    def toggle(self):
        self.open = not self.open
        self.hovered = -1
        self.fan_positions = []

    def close(self):
        self.open = False
        self.hovered = -1
        self.fan_positions = []

    def anchor_to_hand(self, lm, fw, fh):
        """Compute fan card positions from hand landmarks. Updates self.fan_positions."""
        positions = []
        # First 4 items: above each finger MCP
        for i, mcp_idx in enumerate(_FAN_MCP_INDICES):
            if i >= len(ITEMS):
                break
            x = int(lm[mcp_idx].x * fw)
            y = int(lm[mcp_idx].y * fh) - _FAN_OFFSET_Y
            x = max(CARD_W // 2 + 2, min(fw - CARD_W // 2 - 2, x))
            y = max(CARD_H // 2 + 2, min(fh - CARD_H // 2 - 2, y))
            positions.append((x, y))
        # Overflow items: stacked above wrist
        wx = int(lm[0].x * fw)
        wy = int(lm[0].y * fh)
        for i in range(len(_FAN_MCP_INDICES), len(ITEMS)):
            extra = i - len(_FAN_MCP_INDICES)
            x = wx
            y = wy - _FAN_OFFSET_Y - extra * (CARD_H + 8)
            x = max(CARD_W // 2 + 2, min(fw - CARD_W // 2 - 2, x))
            y = max(CARD_H // 2 + 2, min(fh - CARD_H // 2 - 2, y))
            positions.append((x, y))
        self.fan_positions = positions
        return positions

    def update_hover(self, frame_w, frame_h, cursor_screen_x, cursor_screen_y,
                     screen_w, screen_h, hand_lm=None):
        """Map screen cursor position back to frame coordinates for hover detection."""
        fx = int(cursor_screen_x / screen_w * frame_w)
        fy = int(cursor_screen_y / screen_h * frame_h)
        self.hovered = -1

        if hand_lm is not None:
            self.anchor_to_hand(hand_lm, frame_w, frame_h)
            for i, (cx, cy) in enumerate(self.fan_positions):
                if i >= len(ITEMS):
                    break
                if abs(fx - cx) <= CARD_W // 2 and abs(fy - cy) <= CARD_H // 2:
                    self.hovered = i
                    break
        else:
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
        """Compute top-left of menu panel (center of frame) — fallback without hand."""
        total_h = len(ITEMS) * ITEM_H + PADDING * 2
        self._x0 = (frame_w - ITEM_W) // 2
        self._y0 = (frame_h - total_h) // 2 + PADDING
        return self._x0, self._y0
