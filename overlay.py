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

    def draw(self, frame, mode, hand_detected, palm_progress=0.0, menu=None, cursor_pos=None, debug_lm=None, gestures=None, cam_margin=None, mapping_bounds=None):
        fps = self._fps()
        h, w = frame.shape[:2]

        if mapping_bounds is not None:
            self._draw_mapping_area(frame, mapping_bounds)
        elif cam_margin is not None:
            m = cam_margin
            self._draw_mapping_area(frame, (m, 1.0 - m, m, 1.0 - m))

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
            "Q=sair | O=overlay | K=teclado | B=pincel | F=calibrar | ?=gestos",
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

    def _draw_mapping_area(self, frame, bounds):
        """bounds = (min_x, max_x, min_y, max_y) in normalized [0,1] coords."""
        h, w = frame.shape[:2]
        x1 = int(bounds[0] * w)
        x2 = int(bounds[1] * w)
        y1 = int(bounds[2] * h)
        y2 = int(bounds[3] * h)
        aw = x2 - x1
        ah = y2 - y1

        # Dead zone: heavily darken area outside active zone
        buf = frame.copy()
        cv2.rectangle(buf, (0, 0), (w, y1), (0, 0, 0), -1)
        cv2.rectangle(buf, (0, y2), (w, h), (0, 0, 0), -1)
        cv2.rectangle(buf, (0, y1), (x1, y2), (0, 0, 0), -1)
        cv2.rectangle(buf, (x2, y1), (w, y2), (0, 0, 0), -1)
        cv2.addWeighted(buf, 0.6, frame, 0.4, 0, frame)

        # Active zone: subtle vignette so center is visually clear
        inner_buf = frame.copy()
        cv2.rectangle(inner_buf, (x1, y1), (x2, y2), (0, 30, 40), -1)
        cv2.addWeighted(inner_buf, 0.12, frame, 0.88, 0, frame)

        # 3x3 grid — screen quadrant reference
        for col in range(1, 3):
            lx = x1 + col * aw // 3
            cv2.line(frame, (lx, y1 + 1), (lx, y2 - 1), (0, 120, 160), 1)
        for row in range(1, 3):
            ly = y1 + row * ah // 3
            cv2.line(frame, (x1 + 1, ly), (x2 - 1, ly), (0, 120, 160), 1)

        # Corner tick marks (like a monitor bezel)
        tick = 14
        color_corner = (0, 220, 255)
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x2, y2, -1, -1), (x1, y2, 1, -1)]:
            cv2.line(frame, (cx, cy), (cx + dx * tick, cy), color_corner, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy * tick), color_corner, 2)

        # Outer border = screen edges (prominent)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 240), 2)

        # Center label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"TELA  {aw}x{ah}px"
        lw, lh = cv2.getTextSize(label, font, 0.35, 1)[0]
        cv2.putText(frame, label, (x1 + (aw - lw) // 2, y1 + 14),
                    font, 0.35, (0, 180, 210), 1)

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
        py = h - 80
        buf = frame.copy()
        cv2.rectangle(buf, (px - 4, py - 18), (px + 270, py + 42), (10, 10, 10), -1)
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
        # pseudo-depth bar
        depth = math.sqrt((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2)
        depth_norm = max(0.0, min(1.0, (depth - 0.08) / (0.25 - 0.08)))
        bar_w = int(150 * depth_norm)
        cv2.rectangle(frame, (px, py + 28), (px + 150, py + 38), (50, 50, 50), -1)
        cv2.rectangle(frame, (px, py + 28), (px + bar_w, py + 38), (0, 200, 255), -1)
        cv2.putText(frame, f"DEPTH:{depth:.3f}", (px + 156, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

    def draw_calibration(self, frame, calibrator, hand_detected, progress):
        h, w = frame.shape[:2]
        buf = frame.copy()
        cv2.rectangle(buf, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(buf, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "CALIBRACAO", (w // 2 - 110, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2)
        cv2.putText(frame, f"Passo {calibrator.step + 1} de {calibrator.total}",
                    (w // 2 - 70, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        mid_y = h // 2
        cv2.putText(frame, "Mova a mao para o canto:", (w // 2 - 155, mid_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, calibrator.label, (w // 2 - 160, mid_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 255), 2)

        if not hand_detected:
            cv2.putText(frame, "Mostre a mao direita!", (w // 2 - 120, mid_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 1)
        else:
            bx1, by1, bx2, by2 = 60, mid_y + 40, w - 60, mid_y + 58
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
            fill = int((bx2 - bx1) * progress)
            cv2.rectangle(frame, (bx1, by1), (bx1 + fill, by2), (0, 220, 255), -1)

        # Crosshair at target corner
        pad = 36
        tx = pad if calibrator.target_fx == 0 else w - pad
        ty = pad if calibrator.target_fy == 0 else h - pad
        cv2.drawMarker(frame, (tx, ty), (0, 220, 255), cv2.MARKER_CROSS, 32, 2)
        cv2.circle(frame, (tx, ty), 18, (0, 220, 255), 1)

        cv2.putText(frame, "F = pular calibracao", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        return frame

    def show(self, frame):
        cv2.imshow(self._win, frame)

    def toggle(self):
        self.active = not self.active
        if not self.active:
            cv2.destroyWindow(self._win)
