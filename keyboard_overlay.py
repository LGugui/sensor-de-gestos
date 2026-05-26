import cv2
from pynput.keyboard import Controller, Key

_KB = Controller()

# QWERTY layout rows
_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]
_SPECIAL = [
    ("SPACE", " "),
    ("BACK", Key.backspace),
    ("ENTER", Key.enter),
    ("ESC", None),  # closes keyboard
]

KEY_W = 42
KEY_H = 42
KEY_GAP = 4
PADDING = 16


class VirtualKeyboard:
    def __init__(self):
        self.open = False
        self.hovered = None   # (row_type, index) or ("special", index)
        self._keys = []       # list of (label, value, x, y, w, h)

    def toggle(self):
        self.open = not self.open
        self.hovered = None

    def close(self):
        self.open = False
        self.hovered = None

    def _build_layout(self, frame_w, frame_h):
        self._keys = []
        rows = _ROWS
        total_rows = len(rows) + 1  # +1 for special row
        total_h = total_rows * (KEY_H + KEY_GAP) + PADDING * 2
        start_y = frame_h - total_h - 10

        for ri, row in enumerate(rows):
            row_w = len(row) * (KEY_W + KEY_GAP) - KEY_GAP
            start_x = (frame_w - row_w) // 2
            for ci, ch in enumerate(row):
                x = start_x + ci * (KEY_W + KEY_GAP)
                y = start_y + ri * (KEY_H + KEY_GAP)
                self._keys.append((ch, ch, x, y, KEY_W, KEY_H))

        # Special row
        spec_w = len(_SPECIAL) * (KEY_W * 2 + KEY_GAP) - KEY_GAP
        sx = (frame_w - spec_w) // 2
        sy = start_y + len(rows) * (KEY_H + KEY_GAP)
        for i, (label, val) in enumerate(_SPECIAL):
            x = sx + i * (KEY_W * 2 + KEY_GAP)
            self._keys.append((label, val, x, sy, KEY_W * 2, KEY_H))

    def update_hover(self, frame_w, frame_h, cursor_screen_x, cursor_screen_y, screen_w, screen_h):
        fx = int(cursor_screen_x / screen_w * frame_w)
        fy = int(cursor_screen_y / screen_h * frame_h)
        self.hovered = None
        self._build_layout(frame_w, frame_h)
        for i, (label, val, kx, ky, kw, kh) in enumerate(self._keys):
            if kx <= fx <= kx + kw and ky <= fy <= ky + kh:
                self.hovered = i
                break

    def press_hovered(self):
        """Press the currently hovered key. Returns True if keyboard should close."""
        if self.hovered is None:
            return False
        label, val, *_ = self._keys[self.hovered]
        if val is None:  # ESC key
            self.close()
            return True
        _KB.press(val)
        _KB.release(val)
        return False

    def draw(self, frame):
        h, w = frame.shape[:2]
        self._build_layout(w, h)

        # Semi-transparent background
        buf = frame.copy()
        # find bounding box of all keys
        if self._keys:
            xs = [k[2] for k in self._keys]
            ys = [k[3] for k in self._keys]
            x1 = min(xs) - PADDING
            y1 = min(ys) - PADDING
            x2 = max(k[2] + k[4] for k in self._keys) + PADDING
            y2 = max(k[3] + k[5] for k in self._keys) + PADDING
            cv2.rectangle(buf, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.addWeighted(buf, 0.8, frame, 0.2, 0, frame)

        for i, (label, val, kx, ky, kw, kh) in enumerate(self._keys):
            is_hov = (i == self.hovered)
            bg = (240, 240, 240) if is_hov else (60, 60, 60)
            fg = (10, 10, 10) if is_hov else (220, 220, 220)
            cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), bg, -1)
            cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (100, 100, 100), 1)
            font_scale = 0.45 if len(label) > 3 else 0.55
            tx = kx + kw // 2 - len(label) * 6
            ty = ky + kh // 2 + 6
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 1)

        # Title
        cv2.putText(frame, "TECLADO VIRTUAL — Punho=fechar | Pinch=digitar",
                    (10, min(ky - 6 for _, _, _, ky, _, _ in self._keys) if self._keys else 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        return frame
