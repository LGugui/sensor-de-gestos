import time

import cv2

from menu import ITEMS, ITEM_H, ITEM_W, PADDING


class Overlay:
    def __init__(self):
        self.active = True
        self._flash = 0
        self._fps_hist = []
        self._last_t = time.time()
        self._win = "Sensor de Gestos"

    def trigger_flash(self):
        self._flash = 3

    def _fps(self):
        now = time.time()
        dt = max(now - self._last_t, 0.001)
        self._last_t = now
        self._fps_hist.append(1.0 / dt)
        if len(self._fps_hist) > 10:
            self._fps_hist.pop(0)
        return sum(self._fps_hist) / len(self._fps_hist)

    def draw(self, frame, mode, hand_detected, palm_progress=0.0, menu=None, cursor_pos=None):
        fps = self._fps()
        h, w = frame.shape[:2]

        if self._flash > 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 8)
            self._flash -= 1

        cor = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"Modo: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(
            frame,
            "MAO DETECTADA" if hand_detected else "SEM MAO",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2,
        )
        cv2.putText(
            frame,
            "Q=sair | O=overlay | R=recarregar | Palma=menu",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # Palm hold progress bar
        if 0.0 < palm_progress < 1.0:
            bar_w = int((w - 20) * palm_progress)
            cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 20), (0, 200, 255), -1)
            cv2.rectangle(frame, (10, h - 30), (w - 10, h - 20), (100, 100, 100), 1)

        # Menu
        if menu is not None and menu.open:
            self._draw_menu(frame, menu, cursor_pos)

        return frame

    def _draw_menu(self, frame, menu, cursor_pos):
        h, w = frame.shape[:2]
        x0, y0 = menu.anchor(w, h)
        total_h = len(ITEMS) * ITEM_H + PADDING * 2

        # Semi-transparent background
        overlay_buf = frame.copy()
        cv2.rectangle(
            overlay_buf,
            (x0 - PADDING, y0 - PADDING),
            (x0 + ITEM_W + PADDING, y0 + total_h),
            (30, 30, 30), -1,
        )
        cv2.addWeighted(overlay_buf, 0.75, frame, 0.25, 0, frame)

        for i, (label, _) in enumerate(ITEMS):
            iy = y0 + i * ITEM_H
            is_hovered = (i == menu.hovered)

            if is_hovered:
                cv2.rectangle(frame, (x0, iy), (x0 + ITEM_W, iy + ITEM_H - 2), (255, 255, 255), -1)
                text_color = (20, 20, 20)
            else:
                text_color = (220, 220, 220)

            cv2.putText(
                frame, label,
                (x0 + 10, iy + ITEM_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
            )

        # Title bar
        cv2.putText(
            frame, "  MENU",
            (x0, y0 - PADDING + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
        )

    def show(self, frame):
        cv2.imshow(self._win, frame)

    def toggle(self):
        self.active = not self.active
        if not self.active:
            cv2.destroyWindow(self._win)
