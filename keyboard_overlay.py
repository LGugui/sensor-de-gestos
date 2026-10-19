import threading
import tkinter as tk
import tkinter.font as tkfont

import cv2
from pynput.keyboard import Controller, Key

_KB = Controller()

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
    ("ESC", None),
]

KEY_W   = 68
KEY_H   = 50
KEY_GAP = 6
PADDING = 14
_N_ROWS = len(_ROWS) + 1        # letter rows + special row
KB_H    = _N_ROWS * (KEY_H + KEY_GAP) + PADDING * 2 + 24   # +24 for title bar


class VirtualKeyboard:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.open    = False
        self.hovered = None
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._keys   = []
        self._build_layout()

        self._ready  = threading.Event()
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)

    # ── Layout ────────────────────────────────────────────────────────

    def _build_layout(self):
        """Key positions in screen coordinates (bottom of screen)."""
        self._keys = []
        y0 = (self._screen_h - KB_H) // 2 + PADDING + 22   # offset for title text

        for ri, row in enumerate(_ROWS):
            row_w = len(row) * (KEY_W + KEY_GAP) - KEY_GAP
            sx = (self._screen_w - row_w) // 2
            for ci, ch in enumerate(row):
                x = sx + ci * (KEY_W + KEY_GAP)
                y = y0 + ri * (KEY_H + KEY_GAP)
                self._keys.append((ch, ch, x, y, KEY_W, KEY_H))

        spec_w = len(_SPECIAL) * (KEY_W * 2 + KEY_GAP) - KEY_GAP
        sx = (self._screen_w - spec_w) // 2
        sy = y0 + len(_ROWS) * (KEY_H + KEY_GAP)
        for i, (label, val) in enumerate(_SPECIAL):
            x = sx + i * (KEY_W * 2 + KEY_GAP)
            self._keys.append((label, val, x, sy, KEY_W * 2, KEY_H))

    # ── Tkinter thread ────────────────────────────────────────────────

    def _run_tk(self):
        self._root = tk.Tk()
        self._root.overrideredirect(True)          # no title bar / border
        self._root.wm_attributes("-topmost", True) # float above everything
        self._root.wm_attributes("-alpha", 0.90)
        self._root.configure(bg="#080808")
        self._root.geometry(
            f"{self._screen_w}x{KB_H}+0+{(self._screen_h - KB_H) // 2}"
        )
        self._root.withdraw()  # hidden by default

        self._canvas = tk.Canvas(
            self._root,
            width=self._screen_w, height=KB_H,
            bg="#080808", highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        self._font    = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self._font_sm = tkfont.Font(family="Segoe UI", size=9)
        self._font_lg = tkfont.Font(family="Segoe UI", size=14, weight="bold")

        self._prev_open    = False
        self._prev_hovered = -2
        self._ready.set()
        self._root.after(33, self._tk_refresh)
        self._root.mainloop()

    def _tk_refresh(self):
        open_now = self.open
        hov_now  = self.hovered

        # Show / hide window when open state changes
        if open_now != self._prev_open:
            if open_now:
                self._root.deiconify()
                self._root.lift()
                self._root.wm_attributes("-topmost", True)
            else:
                self._root.withdraw()
            self._prev_open    = open_now
            self._prev_hovered = -2   # force redraw

        # Redraw only when hover changes
        if open_now and hov_now != self._prev_hovered:
            self._canvas.delete("all")
            # Title strip
            self._canvas.create_rectangle(
                0, 0, self._screen_w, 20, fill="#0a0a20", outline=""
            )
            self._canvas.create_text(
                self._screen_w // 2, 11,
                text="TECLADO VIRTUAL  —  Blink = digitar  |  Olhos fechados 4s = fechar",
                fill="#0088ff", font=self._font_sm,
            )
            # Keys
            ky_base = (self._screen_h - KB_H) // 2
            for i, (label, val, kx, ky, kw, kh) in enumerate(self._keys):
                hover = (i == hov_now)
                cx = kx
                cy = ky - ky_base
                bg  = "#e0e8ff" if hover else "#1e1e2a"
                fg  = "#0a0a0a" if hover else "#cccccc"
                brd = "#00aaff" if hover else "#333344"
                self._canvas.create_rectangle(
                    cx, cy, cx + kw, cy + kh,
                    fill=bg, outline=brd, width=2 if hover else 1,
                )
                font = self._font_sm if len(label) > 4 else (
                    self._font_lg if len(label) == 1 else self._font
                )
                self._canvas.create_text(
                    cx + kw // 2, cy + kh // 2,
                    text=label, fill=fg, font=font,
                )
            self._prev_hovered = hov_now

        self._root.after(33, self._tk_refresh)

    # ── Public API (called from main thread) ─────────────────────────

    def toggle(self):
        self.open = not self.open
        self.hovered = None

    def close(self):
        self.open = False
        self.hovered = None

    def update_hover(self, frame_w, frame_h,
                     cursor_screen_x, cursor_screen_y,
                     screen_w, screen_h):
        """Map screen cursor position directly to floating keyboard keys."""
        self.hovered = None
        for i, (label, val, kx, ky, kw, kh) in enumerate(self._keys):
            if kx <= cursor_screen_x <= kx + kw and ky <= cursor_screen_y <= ky + kh:
                self.hovered = i
                break

    def press_hovered(self):
        if self.hovered is None:
            return False
        label, val, *_ = self._keys[self.hovered]
        if val is None:   # ESC key
            self.close()
            return True
        _KB.press(val)
        _KB.release(val)
        return False

    def draw(self, frame):
        """Draw small badge on camera frame when keyboard is active."""
        if not self.open:
            return frame
        h, w = frame.shape[:2]
        badge = "[ TECLADO FLUTUANTE ATIVO ]"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx = (w - bw) // 2
        by = 22
        buf = frame.copy()
        cv2.rectangle(buf, (bx - 10, by - bh - 5),
                      (bx + bw + 10, by + 7), (8, 8, 60), -1)
        cv2.addWeighted(buf, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (bx - 10, by - bh - 5),
                      (bx + bw + 10, by + 7), (0, 120, 255), 1)
        cv2.putText(frame, badge, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 180, 255), 1)
        return frame
