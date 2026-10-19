import collections
import math
import time

import cv2

from menu import ITEMS, ITEM_H, ITEM_W, PADDING, CARD_W, CARD_H


class Overlay:
    def __init__(self):
        self.active = True
        self.show_cheatsheet = False
        self.show_debug = False
        self._flash = 0
        self._fps_hist = []
        self._last_t = time.time()
        self._win = "Sensor de Gestos"
        self._cursor_trail = collections.deque(maxlen=12)
        self._ripples = []  # list of (x, y, start_time)

    def trigger_flash(self):
        self._flash = 3

    def trigger_ripple(self, x, y):
        self._ripples.append((x, y, time.time()))

    def _fps(self):
        now = time.time()
        dt = max(now - self._last_t, 0.001)
        self._last_t = now
        self._fps_hist.append(1.0 / dt)
        if len(self._fps_hist) > 10:
            self._fps_hist.pop(0)
        return sum(self._fps_hist) / len(self._fps_hist)

    def draw(self, frame, mode, hand_detected, palm_progress=0.0, menu=None,
             cursor_pos=None, debug_lm=None, gestures=None, cam_margin=None,
             mapping_bounds=None, screen_size=None, touch_enabled=True,
             draw_color=None, watch_progress=0.0,
             cursor_source="MAOS", gaze_calibrated=False,
             face_lm=None, face_blendshapes=None,
             eye_action="", dwell_progress=0.0, gaze_quality=""):
        fps = self._fps()
        h, w = frame.shape[:2]

        if mapping_bounds is not None:
            self._draw_mapping_area(frame, mapping_bounds)
        elif cam_margin is not None:
            m = cam_margin
            self._draw_mapping_area(frame, (m, 1.0 - m, m, 1.0 - m))

        # Movement indicator: cursor position projected back onto camera frame
        if (cursor_pos is not None and mapping_bounds is not None
                and screen_size is not None and hand_detected):
            self._draw_cursor_indicator(frame, cursor_pos, mapping_bounds, screen_size)

        if self._flash > 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 8)
            self._flash -= 1

        cor = (0, 255, 0) if hand_detected else (0, 0, 255)

        # Wrist watch (always drawn when hand present)
        if debug_lm is not None:
            self._draw_wrist_watch(frame, debug_lm, progress=watch_progress)

        # Mode badge: follows wrist when hand detected, else fixed top-left
        if debug_lm is not None:
            self._draw_wrist_badge(frame, mode, hand_detected, debug_lm)
        else:
            cv2.putText(frame, f"Modo: {mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30 if debug_lm is not None else 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        cv2.putText(
            frame,
            "MAO DETECTADA" if hand_detected else "SEM MAO",
            (10, 52 if debug_lm is not None else 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1,
        )
        cv2.putText(
            frame,
            "Q=sair | O=overlay | K=teclado | B=pincel | F=calibrar | T=toques | E=fonte | ?=gestos",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # Calibration quality banner (shown for a few seconds after gaze calibration)
        if gaze_quality:
            _Q_COLORS = {"boa": (0, 200, 80), "razoavel": (20, 200, 255), "ruim": (0, 60, 255)}
            q_col = _Q_COLORS.get(gaze_quality, (180, 180, 180))
            q_label = f"Calibracao: {gaze_quality.upper()}"
            (qlw, qlh), _ = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            qx = (w - qlw) // 2
            qy = 42
            qbuf = frame.copy()
            cv2.rectangle(qbuf, (qx - 12, qy - qlh - 8), (qx + qlw + 12, qy + 8), (8, 8, 8), -1)
            cv2.addWeighted(qbuf, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (qx - 12, qy - qlh - 8), (qx + qlw + 12, qy + 8), q_col, 1)
            cv2.putText(frame, q_label, (qx, qy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, q_col, 2)

        # Cursor source badge (top-right, shown when not using hands)
        if cursor_source != "MAOS":
            src_col = (0, 200, 255) if cursor_source == "ROSTO" else (200, 100, 255)
            src_label = f"[{cursor_source}]"
            if cursor_source == "OLHOS" and not gaze_calibrated:
                src_label += "  G=calibrar"
            (slw, slh), _ = cv2.getTextSize(src_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            sx0 = w - slw - 10
            buf_s = frame.copy()
            cv2.rectangle(buf_s, (sx0 - 6, 6), (w - 4, slh + 14), (10, 10, 10), -1)
            cv2.addWeighted(buf_s, 0.70, frame, 0.30, 0, frame)
            cv2.putText(frame, src_label, (sx0, slh + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, src_col, 2)

        # Face landmarks (iris + eye corners + blink state)
        if face_lm is not None:
            self._draw_face_landmarks(frame, face_lm, face_blendshapes,
                                      cursor_source=cursor_source,
                                      eye_action=eye_action,
                                      dwell_progress=dwell_progress)

        if not touch_enabled:
            # Prominent banner: touches disabled
            bw, bh = 260, 34
            bx, by = w // 2 - bw // 2, 4
            buf = frame.copy()
            cv2.rectangle(buf, (bx, by), (bx + bw, by + bh), (0, 0, 160), -1)
            cv2.addWeighted(buf, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.putText(frame, "TOQUES DESATIVADOS  [ T ]",
                        (bx + 8, by + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 80, 255), 2)

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
            self._draw_menu(frame, menu, cursor_pos, hand_lm=debug_lm)

        # Fingertip contextual indicators
        if debug_lm is not None:
            self._draw_fingertip_indicators(frame, mode, debug_lm, draw_color)

        # Ripple / shockwave rings (on top of everything)
        now = time.time()
        self._ripples = [(rx, ry, rt) for rx, ry, rt in self._ripples if now - rt < 0.3]
        if self._ripples and mapping_bounds is not None and screen_size is not None:
            sw, sh = screen_size
            mb_x1 = int(mapping_bounds[0] * w)
            mb_x2 = int(mapping_bounds[1] * w)
            mb_y1 = int(mapping_bounds[2] * h)
            mb_y2 = int(mapping_bounds[3] * h)
            for rx, ry, rt in self._ripples:
                p = (now - rt) / 0.3          # 0→1 over 300 ms
                a = 1.0 - p
                fx = int(mb_x1 + (rx / sw) * (mb_x2 - mb_x1))
                fy = int(mb_y1 + (ry / sh) * (mb_y2 - mb_y1))
                r_out = int(5 + p * 50)
                r_in  = int(5 + p * 28)
                col_out = (0, int(220 * a), int(255 * a))
                col_in  = (int(200 * a), int(200 * a), int(255 * a))
                cv2.circle(frame, (fx, fy), r_out, col_out, max(1, int(3 * (1 - p))))
                cv2.circle(frame, (fx, fy), r_in,  col_in,  1)

        return frame

    def _draw_cursor_indicator(self, frame, cursor_pos, bounds, screen_size):
        """Project screen cursor position back onto the camera frame within the mapping zone."""
        h, w = frame.shape[:2]
        sw, sh = screen_size
        cx, cy = cursor_pos

        # Map screen coords → frame coords via bounds
        x1 = int(bounds[0] * w)
        x2 = int(bounds[1] * w)
        y1 = int(bounds[2] * h)
        y2 = int(bounds[3] * h)

        fx = int(x1 + (cx / sw) * (x2 - x1))
        fy = int(y1 + (cy / sh) * (y2 - y1))

        self._cursor_trail.append((fx, fy))

        # Draw fade trail
        n = len(self._cursor_trail)
        pts = list(self._cursor_trail)
        for i in range(1, n):
            alpha = i / n
            intensity = int(80 + 120 * alpha)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[i - 1], pts[i], (0, intensity, intensity), thickness)

        # Cursor crosshair
        arm = 10
        cv2.line(frame, (fx - arm, fy), (fx + arm, fy), (0, 0, 0), 3)
        cv2.line(frame, (fx, fy - arm), (fx, fy + arm), (0, 0, 0), 3)
        cv2.line(frame, (fx - arm, fy), (fx + arm, fy), (0, 255, 180), 1)
        cv2.line(frame, (fx, fy - arm), (fx, fy + arm), (0, 255, 180), 1)
        cv2.circle(frame, (fx, fy), 4, (0, 255, 180), -1)
        cv2.circle(frame, (fx, fy), 10, (0, 200, 140), 1)

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

    _WATCH_R = 45  # conservative badge-offset constant (actual band is dynamic)

    def _draw_wrist_watch(self, frame, lm, progress=0.0):
        """Cyberpunk 3D wristband anchored to hand geometry via MediaPipe landmarks."""
        h, w = frame.shape[:2]

        # ── Geometry ────────────────────────────────────────────────────
        cx = int(lm[0].x * w)
        cy = int(lm[0].y * h)

        # Band major axis: pinky-MCP(17) → index-MCP(5)
        ax = (lm[5].x - lm[17].x) * w
        ay = (lm[5].y - lm[17].y) * h
        axis_len = math.sqrt(ax * ax + ay * ay) or 1
        R = int(max(22, min(82, axis_len * 0.55)))
        ang = math.degrees(math.atan2(ay, ax))

        # Minor axis: cross-product with up-vector (wrist→middle-MCP) gives foreshortening
        ux = lm[9].x * w - cx
        uy = lm[9].y * h - cy
        u_len = math.sqrt(ux * ux + uy * uy) or 1
        sin_t = abs(ax / axis_len * uy / u_len - ay / axis_len * ux / u_len)
        b = int(max(6, R * (0.12 + 0.32 * sin_t)))

        # Band ring thickness
        t = max(8, R // 4)

        # Safe center
        cx = max(R + t + 5, min(w - R - t - 5, cx))
        cy = max(b + t + 5, min(h - b - t - 5, cy))
        ctr = (cx, cy)

        # "Near-side" unit vector: perpendicular to band axis, pointing toward fingers
        px1, py1 = -ay / axis_len,  ax / axis_len
        px2, py2 =  ay / axis_len, -ax / axis_len
        near_px = px1 if (px1 * ux / u_len + py1 * uy / u_len) > 0 else px2
        near_py = py1 if (px1 * ux / u_len + py1 * uy / u_len) > 0 else py2

        # ── Palette (BGR) ────────────────────────────────────────────────
        C_NEON   = (0,  255, 185)
        C_DIM    = (0,   90,  68)
        C_DARK   = (4,    8,  14)
        C_ACC    = (200,  12, 210)
        C_BRIGHT = (210, 255, 248)
        C_PROG   = (0,  242, 255)

        # ── 1. Outer ambient glow (blended) ────────────────────────────
        g_buf = frame.copy()
        for goff, alpha in [(12, 0.18), (7, 0.28), (3, 0.40)]:
            gc = (int(C_NEON[0]*alpha), int(C_NEON[1]*alpha), int(C_NEON[2]*alpha))
            cv2.ellipse(g_buf, ctr, (R + goff, b + goff // 2), ang, 0, 360, gc, 2)
        cv2.addWeighted(g_buf, 0.55, frame, 0.45, 0, frame)

        # ── 2. Dark band ring (thick outline = donut) ──────────────────
        cv2.ellipse(frame, ctr, (R, b), ang, 0, 360, C_DARK, t + 3)

        # ── 3. Segmented neon arcs with depth shading ──────────────────
        N_SEG = 16
        seg_deg = 360.0 / N_SEG
        GAP = 7.0
        for i in range(N_SEG):
            sa = i * seg_deg + GAP / 2
            ea = (i + 1) * seg_deg - GAP / 2
            mid_abs = ang + sa + (ea - sa) / 2
            seg_nx = math.cos(math.radians(mid_abs))
            seg_ny = math.sin(math.radians(mid_abs))
            shade = 0.40 + 0.60 * max(0.0, seg_nx * near_px + seg_ny * near_py)

            if i % 4 == 0:                              # accent every 4th
                col = tuple(int(c * shade) for c in C_ACC)
                thick = 2
            elif i % 4 == 2:                            # highlight every other
                col = tuple(int(c * shade) for c in C_BRIGHT)
                thick = 1
            else:
                col = tuple(int(c * shade) for c in C_NEON)
                thick = 2
            cv2.ellipse(frame, ctr, (R, b), ang, sa, ea, col, thick)

        # ── 4. Inner ring highlight ─────────────────────────────────────
        cv2.ellipse(frame, ctr,
                    (max(2, R - t // 2), max(2, b - t // 4)),
                    ang, 0, 360, C_DIM, 1)

        # ── 5. Cardinal LED dots (4 × 90°) ─────────────────────────────
        for i, deg_off in enumerate([0, 90, 180, 270]):
            rad = math.radians(ang + deg_off)
            dp = (int(cx + R * math.cos(rad)), int(cy + b * math.sin(rad)))
            col = C_ACC if deg_off in (90, 270) else C_BRIGHT
            cv2.circle(frame, dp, 3, col, -1)
            cv2.circle(frame, dp, 5, col, 1)

        # ── 6. Cyberpunk display chip (near-side center) ────────────────
        chip_r = max(4, R // 7)
        chip_x = int(cx + near_px * (b + t // 2 + chip_r + 2))
        chip_y = int(cy + near_py * (b + t // 2 + chip_r + 2))
        chip_x = max(chip_r + 3, min(w - chip_r - 3, chip_x))
        chip_y = max(chip_r + 3, min(h - chip_r - 3, chip_y))
        cv2.circle(frame, (chip_x, chip_y), chip_r + 3, C_DARK, -1)
        cv2.circle(frame, (chip_x, chip_y), chip_r,     C_ACC,  -1)
        cv2.circle(frame, (chip_x, chip_y), chip_r + 3, C_ACC,   1)
        # Tiny circuit traces from chip
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1)]:
            cv2.line(frame,
                     (chip_x + ddx * (chip_r + 3), chip_y + ddy * (chip_r + 3)),
                     (chip_x + ddx * (chip_r + 9), chip_y + ddy * (chip_r + 9)),
                     C_DIM, 1)
        # Blinking dot on chip center (fast pulse from time)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (chip_x, chip_y), max(1, chip_r // 2), C_BRIGHT, -1)

        # ── 7. Hold-progress arc (tulip gesture) ──────────────────────
        if progress > 0.0:
            sweep = int(360 * progress)
            p_buf = frame.copy()
            cv2.ellipse(p_buf, ctr,
                        (R + t // 2 + 5, b + t // 4 + 4),
                        ang, -90, -90 + sweep, C_PROG, 8)
            cv2.addWeighted(p_buf, 0.60, frame, 0.40, 0, frame)
            cv2.ellipse(frame, ctr,
                        (R + t // 2 + 2, b + t // 4 + 1),
                        ang, -90, -90 + sweep, C_PROG, 3)
            cv2.ellipse(frame, ctr,
                        (R + t // 2 + 2, b + t // 4 + 1),
                        ang, -90, -90 + sweep, C_BRIGHT, 1)

    def _draw_wrist_badge(self, frame, mode, hand_detected, lm):
        h, w = frame.shape[:2]
        wx = int(lm[0].x * w)
        wy = int(lm[0].y * h)
        bx = max(4, min(w - 150, wx + 12))
        by = max(22, min(h - 8, wy - self._WATCH_R - 14))

        cor = (0, 255, 0) if hand_detected else (0, 80, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(mode, font, 0.55, 1)
        pad = 5
        x1, y1 = bx - pad, by - th - pad
        x2, y2 = bx + tw + pad, by + pad

        buf = frame.copy()
        cv2.rectangle(buf, (x1, y1), (x2, y2), (10, 10, 10), -1)
        cv2.addWeighted(buf, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 1)
        cv2.putText(frame, mode, (bx, by), font, 0.55, cor, 1)
        # Thin connector from badge to wrist
        cv2.line(frame, (bx + tw // 2, y2), (wx, wy),
                 (0, 80, 0) if hand_detected else (0, 0, 60), 1)

    @staticmethod
    def _lm_visible(lm, idx, threshold=0.5):
        vis = getattr(lm[idx], 'visibility', None)
        if vis is not None and vis >= 0:
            return vis >= threshold
        return lm[idx].z < 0.15

    def _draw_fingertip_indicators(self, frame, mode, lm, draw_color=None):
        h, w = frame.shape[:2]

        def pt(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        _GRAY = (80, 80, 80)

        if "SCROLL" in mode:
            ix, iy = pt(8)
            mx, my = pt(12)
            # Index tip ↑
            col_i = (0, 220, 255) if self._lm_visible(lm, 8) else _GRAY
            cv2.arrowedLine(frame, (ix, iy + 22), (ix, iy - 22),
                            col_i, 2, tipLength=0.35)
            # Middle tip ↓
            col_m = (0, 220, 255) if self._lm_visible(lm, 12) else _GRAY
            cv2.arrowedLine(frame, (mx, my - 22), (mx, my + 22),
                            col_m, 2, tipLength=0.35)

        elif "DRAW" in mode:
            ix, iy = pt(8)
            if self._lm_visible(lm, 8):
                swatch = draw_color if draw_color else (255, 255, 255)
                if isinstance(swatch, str):
                    s = swatch.lstrip("#")
                    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
                    swatch = (b, g, r)
                cv2.circle(frame, (ix, iy - 20), 9, swatch, -1)
                cv2.circle(frame, (ix, iy - 20), 9, (0, 0, 0), 1)
            else:
                cv2.circle(frame, (ix, iy - 20), 9, _GRAY, -1)
            px, py = pt(20)
            col_p = (160, 160, 160) if self._lm_visible(lm, 20) else _GRAY
            cv2.putText(frame, "Z", (px - 5, py - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col_p, 1)

        elif "PRECISAO" in mode:
            tx, ty = pt(4)
            ix, iy = pt(8)
            col = (0, 255, 180) if (self._lm_visible(lm, 4)
                                    and self._lm_visible(lm, 8)) else _GRAY
            mx_c = (tx + ix) // 2
            my_c = (ty + iy) // 2
            arm = 10
            cv2.line(frame, (mx_c - arm, my_c), (mx_c + arm, my_c), col, 2)
            cv2.line(frame, (mx_c, my_c - arm), (mx_c, my_c + arm), col, 2)
            cv2.circle(frame, (mx_c, my_c), 4, col, 1)

    def _draw_face_landmarks(self, frame, face_lm, face_blendshapes=None,
                             cursor_source="MAOS", eye_action="", dwell_progress=0.0):
        """Draw iris centers, eye corners, blink indicators, dwell ring, and action status."""
        h, w = frame.shape[:2]

        def pt(idx):
            return (int(face_lm[idx].x * w), int(face_lm[idx].y * h))

        # Eye corners — small green dots
        for idx in [33, 133, 362, 263]:
            if idx < len(face_lm):
                cv2.circle(frame, pt(idx), 3, (0, 180, 80), -1)

        # Iris rings (468-472 = left, 473-477 = right)
        for idx in range(469, 473):
            if idx < len(face_lm):
                cv2.circle(frame, pt(idx), 2, (0, 160, 220), -1)
        for idx in range(474, 478):
            if idx < len(face_lm):
                cv2.circle(frame, pt(idx), 2, (0, 160, 220), -1)

        # Iris centers — bright cyan
        if len(face_lm) <= 473:
            return

        lc = pt(468)
        rc = pt(473)
        for c in (lc, rc):
            cv2.circle(frame, c, 5, (0, 255, 255), -1)
            cv2.circle(frame, c, 8, (0, 255, 255), 1)

        # Average midpoint crosshair (used by gaze mapper)
        mx = (lc[0] + rc[0]) // 2
        my = (lc[1] + rc[1]) // 2
        arm = 9
        cv2.line(frame, (mx - arm, my), (mx + arm, my), (200, 80, 255), 1)
        cv2.line(frame, (mx, my - arm), (mx, my + arm), (200, 80, 255), 1)
        cv2.circle(frame, (mx, my), 3, (200, 80, 255), -1)

        # Dwell progress ring (OLHOS mode only)
        if cursor_source == "OLHOS" and dwell_progress > 0.02:
            sweep = int(360 * dwell_progress)
            dwell_col = (0, 200, 255) if dwell_progress < 0.9 else (0, 255, 80)
            cv2.ellipse(frame, (mx, my), (16, 16), -90, 0, sweep, dwell_col, 3)

        # Action status label near iris midpoint
        if eye_action:
            is_click = "clique" in eye_action.lower()
            act_col = (0, 255, 80) if is_click else (0, 220, 255)
            (aw, ah), _ = cv2.getTextSize(eye_action, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ax = max(4, min(w - aw - 4, mx - aw // 2))
            ay = max(14, my - 24)
            buf = frame.copy()
            cv2.rectangle(buf, (ax - 4, ay - ah - 3), (ax + aw + 4, ay + 3), (8, 8, 8), -1)
            cv2.addWeighted(buf, 0.72, frame, 0.28, 0, frame)
            cv2.putText(frame, eye_action, (ax, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, act_col, 1)

        # Blink score labels
        if face_blendshapes:
            bs_map = {b.category_name: b.score for b in face_blendshapes}
            bl = bs_map.get("eyeBlinkLeft", 0.0)
            br = bs_map.get("eyeBlinkRight", 0.0)
            col_l = (0, 60, 255) if bl > 0.5 else (0, 200, 80)
            col_r = (0, 60, 255) if br > 0.5 else (0, 200, 80)
            cv2.putText(frame, f"L{bl:.1f}", (lc[0] - 14, lc[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, col_l, 1)
            cv2.putText(frame, f"R{br:.1f}", (rc[0] - 14, rc[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, col_r, 1)

    def _draw_fan_cards(self, frame, menu):
        h, w = frame.shape[:2]
        buf = frame.copy()

        for i, (cx, cy) in enumerate(menu.fan_positions):
            if i >= len(ITEMS):
                break
            is_hov = (i == menu.hovered)
            bg = (220, 220, 220) if is_hov else (20, 30, 40)
            x1, y1 = cx - CARD_W // 2, cy - CARD_H // 2
            x2, y2 = cx + CARD_W // 2, cy + CARD_H // 2
            cv2.rectangle(buf, (x1, y1), (x2, y2), bg, -1)

        cv2.addWeighted(buf, 0.78, frame, 0.22, 0, frame)

        for i, (cx, cy) in enumerate(menu.fan_positions):
            if i >= len(ITEMS):
                break
            label, _ = ITEMS[i]
            is_hov = (i == menu.hovered)
            x1, y1 = cx - CARD_W // 2, cy - CARD_H // 2
            x2, y2 = cx + CARD_W // 2, cy + CARD_H // 2
            border = (0, 240, 255) if is_hov else (0, 130, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if is_hov else 1)
            fg = (10, 10, 10) if is_hov else (210, 210, 210)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
            cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), font, 0.38, fg, 1)
            # Finger index dot
            cv2.circle(frame, (x1 + 8, y1 + 8), 4, border, -1)
            cv2.putText(frame, str(i + 1), (x1 + 5, y1 + 11),
                        font, 0.28, (0, 0, 0), 1)

    def _draw_menu(self, frame, menu, cursor_pos, hand_lm=None):
        # Fan mode: hand landmarks available and positions already computed
        if hand_lm is not None and menu.fan_positions:
            self._draw_fan_cards(frame, menu)
            return

        # Fallback: classic vertical panel (no hand or positions not ready)
        h, w = frame.shape[:2]
        x0, y0 = menu.anchor(w, h)
        total_h = len(ITEMS) * ITEM_H + PADDING * 2

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
            ("5 dedos juntos (tulipa) → menu", (200, 200, 200)),
            ("Relogio com mao ctrl  → menu", (200, 200, 200)),
            ("", None),
            ("OLHO / ROSTO (eye_enabled:true)", (0, 220, 255)),
            ("E  →  ciclar fonte (MAOS/ROSTO/OLHOS)", (200, 200, 200)),
            ("G  →  calibrar olhar (9 pontos)", (200, 200, 200)),
            ("OLHOS: blink=click, dwell=auto-click", (200, 200, 200)),
            ("OLHOS+pinch = click na posicao do olhar", (200, 200, 200)),
            ("ROSTO: inclinar cabeca = scroll", (200, 200, 200)),
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
            ("T  →  toggle toques (mostra mao, sem acao)", (200, 200, 200)),
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

    def draw_calibration(self, frame, calibrator, hand_detected, progress, face_lm=None):
        h, w = frame.shape[:2]
        is_gaze_cal = face_lm is not None or (not hand_detected and face_lm is None
                                              and hasattr(calibrator, '_data'))
        buf = frame.copy()
        cv2.rectangle(buf, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(buf, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "CALIBRACAO", (w // 2 - 110, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2)
        cv2.putText(frame, f"Passo {calibrator.step + 1} de {calibrator.total}",
                    (w // 2 - 70, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        mid_y = h // 2
        instr = "Olhe para o ponto:" if face_lm is not None else "Mova a mao para o canto:"
        cv2.putText(frame, instr, (w // 2 - 155, mid_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, calibrator.label, (w // 2 - 160, mid_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 220, 255), 2)

        if not hand_detected:
            msg = "Rosto nao detectado!" if face_lm is None else ""
            if msg:
                cv2.putText(frame, msg, (w // 2 - 120, mid_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 1)
        else:
            bx1, by1, bx2, by2 = 60, mid_y + 40, w - 60, mid_y + 58
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
            fill = int((bx2 - bx1) * progress)
            cv2.rectangle(frame, (bx1, by1), (bx1 + fill, by2), (0, 220, 255), -1)

        # Target crosshair — works for any (fx, fy) including center 0.5
        pad = 36
        tx = int(pad + calibrator.target_fx * (w - 2 * pad))
        ty = int(pad + calibrator.target_fy * (h - 2 * pad))
        cv2.drawMarker(frame, (tx, ty), (0, 220, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame, (tx, ty), 22, (0, 220, 255), 1)
        # Pulsing ring
        r_pulse = 22 + int(6 * abs(math.sin(time.time() * 4)))
        cv2.circle(frame, (tx, ty), r_pulse, (0, 120, 140), 1)

        # Iris visualization during gaze calibration
        if face_lm is not None and len(face_lm) > 473:
            def gpt(idx): return (int(face_lm[idx].x * w), int(face_lm[idx].y * h))
            lc = gpt(468)
            rc = gpt(473)
            for c in (lc, rc):
                cv2.circle(frame, c, 6, (0, 255, 255), -1)
                cv2.circle(frame, c, 10, (0, 255, 255), 1)
            mx = (lc[0] + rc[0]) // 2
            my = (lc[1] + rc[1]) // 2
            arm = 12
            cv2.line(frame, (mx - arm, my), (mx + arm, my), (200, 80, 255), 2)
            cv2.line(frame, (mx, my - arm), (mx, my + arm), (200, 80, 255), 2)
            cv2.circle(frame, (mx, my), 4, (200, 80, 255), -1)
            cv2.putText(frame, "OLHAR", (mx - 20, my + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 80, 255), 1)
            # Line from iris to target
            cv2.line(frame, (mx, my), (tx, ty), (0, 80, 80), 1)

        cv2.putText(frame, "F = pular calibracao", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        return frame

    def show(self, frame):
        cv2.imshow(self._win, frame)

    def toggle(self):
        self.active = not self.active
        if not self.active:
            cv2.destroyWindow(self._win)
