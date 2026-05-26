import math
import time

import cv2

from menu import ITEMS, ITEM_H, ITEM_W, PADDING


class Overlay:
    def __init__(self):
        self.active = True
        self.show_cheatsheet = False
        self.show_debug = False
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

    def draw(self, frame, mode, hand_detected, palm_progress=0.0, menu=None, cursor_pos=None, debug_lm=None, gestures=None, cam_margin=None):
        fps = self._fps()
        h, w = frame.shape[:2]

        if cam_margin is not None:
            self._draw_mapping_area(frame, cam_margin)

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
            "Q=sair | O=overlay | K=teclado | B=pincel | R=reload | ?=gestos",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        if self.show_cheatsheet:
            self._draw_cheatsheet(frame)

        if self.show_debug and debug_lm is not None and gestures is not None:
            self._draw_debug(frame, debug_lm, gestures)

        # Palm hold progress bar
        if 0.0 < palm_progress < 1.0:
            bar_w = int((w - 20) * palm_progress)
            cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 20), (0, 200, 255), -1)
            cv2.rectangle(frame, (10, h - 30), (w - 10, h - 20), (100, 100, 100), 1)

        # Menu
        if menu is not None and menu.open:
            self._draw_menu(frame, menu, cursor_pos)

        return frame

    def _draw_mapping_area(self, frame, margin):
        h, w = frame.shape[:2]
        m = margin
        x1 = int(m * w)
        y1 = int(m * h)
        x2 = int((1 - m) * w)
        y2 = int((1 - m) * h)

        # Darken areas outside the active zone
        buf = frame.copy()
        cv2.rectangle(buf, (0, 0), (w, y1), (0, 0, 0), -1)        # top
        cv2.rectangle(buf, (0, y2), (w, h), (0, 0, 0), -1)        # bottom
        cv2.rectangle(buf, (0, y1), (x1, y2), (0, 0, 0), -1)      # left
        cv2.rectangle(buf, (x2, y1), (w, y2), (0, 0, 0), -1)      # right
        cv2.addWeighted(buf, 0.45, frame, 0.55, 0, frame)

        # Active zone border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 1)

        # 3x3 grid (each cell = 1/3 of screen)
        aw = x2 - x1
        ah = y2 - y1
        for col in range(1, 3):
            lx = x1 + col * aw // 3
            cv2.line(frame, (lx, y1), (lx, y2), (0, 140, 180), 1)
        for row in range(1, 3):
            ly = y1 + row * ah // 3
            cv2.line(frame, (x1, ly), (x2, ly), (0, 140, 180), 1)

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

    def _draw_cheatsheet(self, frame):
        h, w = frame.shape[:2]
        lines = [
            ("GESTOS — MAO DIREITA (cursor)", (0, 220, 255)),
            ("Indicador esticado  →  mover cursor", (200, 200, 200)),
            ("Pinch polegar+indic →  clique esq", (200, 200, 200)),
            ("Pinch polegar+medio →  clique dir", (200, 200, 200)),
            ("Semi-pinch          →  modo precisao", (200, 200, 200)),
            ("Indic+medio eretos  →  scroll", (200, 200, 200)),
            ("Anelar sozinho (4s) →  modo desenho", (200, 200, 200)),
            ("", None),
            ("GESTOS BIMANUAL", (0, 220, 255)),
            ("Ambas palmas (1.5s) →  menu", (200, 200, 200)),
            ("Punho esq           →  teclado virtual", (200, 200, 200)),
            ("Mindinho sozinho    →  teclado virtual", (200, 200, 200)),
            ("Punho esq (desenho) →  limpar canvas", (200, 200, 200)),
            ("Indicadores tocam   →  carimbar forma", (200, 200, 200)),
            ("", None),
            ("TECLADO", (0, 220, 255)),
            ("K  →  teclado virtual", (200, 200, 200)),
            ("B  →  modo pincel/desenho", (200, 200, 200)),
            ("S  →  ciclar forma (circulo/quadrado...)", (200, 200, 200)),
            ("C  →  ciclar cor do pincel", (200, 200, 200)),
            ("Z  →  desfazer ultimo traco", (200, 200, 200)),
            ("E  →  exportar canvas como PNG", (200, 200, 200)),
            ("?  →  toggle este painel", (200, 200, 200)),
            ("O  →  toggle overlay", (200, 200, 200)),
            ("R  →  recarregar config", (200, 200, 200)),
            ("Q  →  sair", (200, 200, 200)),
        ]
        px, py, pw = w - 360, 10, 350
        ph = len(lines) * 22 + 20
        buf = frame.copy()
        cv2.rectangle(buf, (px - 8, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.addWeighted(buf, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px - 8, py), (px + pw, py + ph), (0, 220, 255), 1)
        for i, (text, color) in enumerate(lines):
            if color and text:
                cv2.putText(frame, text, (px, py + 18 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_debug(self, frame, lm, gestures):
        h, w = frame.shape[:2]
        states = gestures.finger_states(lm)
        names = ["thumb", "index", "middle", "ring", "pinky"]
        labels = ["POL", "IND", "MED", "ANE", "MIN"]
        px = 10
        py = h - 60
        buf = frame.copy()
        cv2.rectangle(buf, (px - 4, py - 18), (px + 270, py + 22), (10, 10, 10), -1)
        cv2.addWeighted(buf, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "DEBUG DEDOS:", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        for i, (name, label) in enumerate(zip(names, labels)):
            on = states[name]
            color = (0, 255, 80) if on else (80, 80, 80)
            cv2.putText(frame, label, (px + 100 + i * 35, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        # pinch distances
        d_left = math.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
        d_right = math.sqrt((lm[4].x - lm[12].x)**2 + (lm[4].y - lm[12].y)**2)
        cv2.putText(frame, f"pinch L:{d_left:.3f} R:{d_right:.3f}", (px, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 80), 1)

    def show(self, frame):
        cv2.imshow(self._win, frame)

    def toggle(self):
        self.active = not self.active
        if not self.active:
            cv2.destroyWindow(self._win)
