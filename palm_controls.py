import math
import time

import cv2

# D-pad zone names
ZONE_CENTER = "CENTER"
ZONE_UP     = "UP"
ZONE_DOWN   = "DOWN"
ZONE_LEFT   = "LEFT"
ZONE_RIGHT  = "RIGHT"

_ZONE_COLORS = {
    ZONE_CENTER: (200, 200, 200),
    ZONE_UP:     (100, 220, 255),
    ZONE_DOWN:   (100, 220, 255),
    ZONE_LEFT:   (100, 220, 255),
    ZONE_RIGHT:  (100, 220, 255),
}
_ACTIVE_COLOR = (0, 255, 200)
_ZONE_RADIUS_PX = 22
_ZONE_LABEL_OFFSET = 28


class PalmControls:
    """D-pad control zones projected onto the control hand's palm.
    The cursor hand's index tip activates zones → scroll/navigation.

    Zones are computed each frame from the control hand landmarks so they
    move with the hand naturally.
    """

    def __init__(self, config):
        self._zone_radius = config.get("palm_zone_radius", 0.06)
        self._debounce = config.get("palm_zone_debounce_frames", 8)
        self._scroll_speed = config.get("scroll_speed", 3)
        self._enabled = config.get("palm_controls_enabled", True)

        self._hover_zone = None
        self._hover_frames = 0
        self._last_fired_zone = None
        self._fire_cooldown = 0.35  # seconds between repeated fires
        self._last_fire_t = 0.0

    # ── Zone geometry ──────────────────────────────────────────────────

    def _compute_zones(self, lm, frame_w, frame_h):
        """Return dict zone_name → (px, py) pixel position."""
        # Palm axis: wrist (0) → middle MCP (9)
        wx, wy = lm[0].x, lm[0].y
        mx, my = lm[9].x, lm[9].y

        # Center = 55% along wrist→MCP
        cx = wx + (mx - wx) * 0.55
        cy = wy + (my - wy) * 0.55

        # Forward unit vector (wrist → MCP)
        dx, dy = mx - wx, my - wy
        length = math.sqrt(dx * dx + dy * dy) or 1e-6
        ux, uy = dx / length, dy / length

        # Perpendicular (right-hand rule in image coords)
        px, py = -uy, ux

        step_fwd  = 0.12
        step_side = 0.10

        zones = {
            ZONE_CENTER: (cx, cy),
            ZONE_UP:     (cx + ux * step_fwd,  cy + uy * step_fwd),
            ZONE_DOWN:   (cx - ux * step_fwd,  cy - uy * step_fwd),
            ZONE_LEFT:   (cx - px * step_side, cy - py * step_side),
            ZONE_RIGHT:  (cx + px * step_side, cy + py * step_side),
        }
        return {name: (int(nx * frame_w), int(ny * frame_h))
                for name, (nx, ny) in zones.items()}

    # ── Detection ──────────────────────────────────────────────────────

    def update(self, cursor_lm, control_lm, frame_w, frame_h):
        """Call every frame. Returns active zone name or None.
        cursor_lm: dominant/cursor hand landmarks (tip = lm[8])
        control_lm: non-dominant hand landmarks (palm = D-pad surface)
        """
        if not self._enabled or control_lm is None or cursor_lm is None:
            self._hover_zone = None
            self._hover_frames = 0
            return None

        zones = self._compute_zones(control_lm, frame_w, frame_h)

        # Cursor index tip in normalized coords
        tip_x = cursor_lm[8].x * frame_w
        tip_y = cursor_lm[8].y * frame_h

        # Find closest zone within threshold radius (pixels)
        radius_px = self._zone_radius * frame_w
        closest_zone = None
        closest_dist = float("inf")
        for name, (zx, zy) in zones.items():
            d = math.sqrt((tip_x - zx) ** 2 + (tip_y - zy) ** 2)
            if d < radius_px and d < closest_dist:
                closest_dist = d
                closest_zone = name

        if closest_zone == self._hover_zone:
            self._hover_frames += 1
        else:
            self._hover_zone = closest_zone
            self._hover_frames = 0

        if self._hover_frames >= self._debounce:
            now = time.time()
            if now - self._last_fire_t >= self._fire_cooldown:
                self._last_fire_t = now
                return self._hover_zone

        return None

    def get_zones(self, control_lm, frame_w, frame_h):
        """Return zone pixel positions for rendering."""
        if control_lm is None:
            return {}
        return self._compute_zones(control_lm, frame_w, frame_h)

    def get_hover_zone(self):
        return self._hover_zone

    def get_hover_progress(self):
        return min(1.0, self._hover_frames / max(1, self._debounce))

    # ── Rendering ──────────────────────────────────────────────────────

    def draw(self, frame, control_lm):
        if not self._enabled or control_lm is None:
            return frame

        h, w = frame.shape[:2]
        zones = self._compute_zones(control_lm, w, h)
        hover = self._hover_zone
        progress = self.get_hover_progress()

        overlay = frame.copy()

        for name, (zx, zy) in zones.items():
            is_active = (name == hover)
            base_color = _ACTIVE_COLOR if is_active else _ZONE_COLORS[name]

            # Filled semi-transparent circle
            cv2.circle(overlay, (zx, zy), _ZONE_RADIUS_PX, base_color, -1)

            # Progress ring when hovering
            if is_active and progress > 0:
                angle = int(360 * progress)
                axes = (_ZONE_RADIUS_PX + 5, _ZONE_RADIUS_PX + 5)
                cv2.ellipse(overlay, (zx, zy), axes, -90, 0, angle,
                            (255, 255, 0), 2)

            # Outer ring
            cv2.circle(overlay, (zx, zy), _ZONE_RADIUS_PX, base_color, 1)

            # Zone label
            label = name[0] if name != ZONE_CENTER else "●"
            cv2.putText(overlay, label,
                        (zx - 5, zy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0) if not is_active else (0, 0, 0), 1)

        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        return frame
